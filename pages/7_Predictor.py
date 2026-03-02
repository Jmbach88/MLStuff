import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import text

from db import get_local_engine, init_local_db
from index import load_index
from predictor import evaluate_case

st.set_page_config(page_title="Case Evaluator", page_icon="⚖️", layout="wide")
st.title("Case Evaluation & Outcome Prediction")
st.info(
    "For research and prioritization only. Not legal advice. "
    "Always review underlying precedents."
)


@st.cache_resource
def get_db_engine():
    engine = get_local_engine()
    init_local_db(engine)
    return engine


@st.cache_resource
def get_faiss_index():
    index, chunk_map = load_index()
    return index, chunk_map


engine = get_db_engine()
index, chunk_map = get_faiss_index()

if index is None or chunk_map is None:
    st.warning("FAISS index not found. Run `python pipeline.py` first.")
    st.stop()

# --- Helper formatters ---

def fmt_dollar(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "N/A"
    return f"${v:,.0f}"


def fmt_pct(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "N/A"
    return f"{v * 100:.1f}%"


# --- Sidebar Filters ---
with st.sidebar:
    st.header("Filters")

    statute_options = ["All", "FDCPA", "TCPA", "FCRA"]
    selected_statute = st.selectbox("Statute", statute_options)

    with engine.connect() as conn:
        circuits = sorted([
            r[0] for r in conn.execute(
                text("SELECT DISTINCT circuit FROM opinions "
                     "WHERE circuit IS NOT NULL AND circuit != ''")
            ).fetchall()
        ])
    selected_circuit = st.selectbox("Circuit", ["All"] + circuits)

    court_types = ["All", "district", "circuit"]
    selected_court_type = st.selectbox("Court Type", court_types)

    top_k = st.slider("Similar Cases (top_k)", min_value=10, max_value=100, value=50)

# Resolve filter args
f_statute = selected_statute if selected_statute != "All" else None
f_circuit = selected_circuit if selected_circuit != "All" else None
f_court = selected_court_type if selected_court_type != "All" else None

# --- Input Section ---
st.subheader("Case Summary")
query_text = st.text_area(
    "Enter case facts and claims",
    height=200,
    placeholder=(
        "Describe the case facts, claims asserted, and key issues. "
        "For example: Plaintiff alleges defendant debt collector called "
        "plaintiff's cell phone using an automatic telephone dialing system "
        "without prior express consent, in violation of the TCPA..."
    ),
)

evaluate_clicked = st.button(
    "Evaluate Case", type="primary", disabled=not query_text
)

if evaluate_clicked and query_text:
    with st.spinner("Searching similar precedents and computing predictions..."):
        result = evaluate_case(
            engine, index, chunk_map, query_text,
            statute=f_statute, circuit=f_circuit,
            court_type=f_court, top_k=top_k,
        )

    if result is None or result["predicted_outcome"] is None:
        st.warning("No similar cases found for the given filters. Try broadening your search.")
        st.stop()

    # --- Summary Metrics Row ---
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        outcome_display = (result["predicted_outcome"] or "Unknown").replace("_", " ").title()
        confidence_display = f"{result['confidence'] * 100:.1f}%"
        st.metric("Predicted Outcome", outcome_display, help=f"Confidence: {confidence_display}")
        st.caption(f"Confidence: {confidence_display}")

    with col2:
        dmg_median = fmt_dollar(result["damages_median"])
        dmg_range = ""
        if result["damages_25th"] is not None and result["damages_75th"] is not None:
            dmg_range = f"{fmt_dollar(result['damages_25th'])} - {fmt_dollar(result['damages_75th'])}"
        st.metric("Est. Damages (Median)", dmg_median)
        if dmg_range:
            st.caption(f"25th-75th: {dmg_range}")

    with col3:
        fee_median = fmt_dollar(result["fees_median"])
        fee_range = ""
        if result["fees_25th"] is not None and result["fees_75th"] is not None:
            fee_range = f"{fmt_dollar(result['fees_25th'])} - {fmt_dollar(result['fees_75th'])}"
        st.metric("Est. Attorney Fees (Median)", fee_median)
        if fee_range:
            st.caption(f"25th-75th: {fee_range}")

    with col4:
        st.metric("Similar Cases Found", result["case_count"])

    # --- Tabs ---
    tab_precedents, tab_risk, tab_claims = st.tabs([
        "Similar Precedents", "Risk Factors", "Claim Recommendations",
    ])

    # =====================================================================
    # Tab 1 — Similar Precedents
    # =====================================================================
    with tab_precedents:
        st.subheader("Top Similar Precedents")

        if not result["similar_cases"]:
            st.info("No similar cases found.")
        else:
            cases = result["similar_cases"][:20]
            df_cases = pd.DataFrame(cases)

            # Format columns for display
            display_df = df_cases[["title", "circuit", "date_issued",
                                   "predicted_outcome", "damages",
                                   "attorney_fees", "similarity_score"]].copy()

            display_df["predicted_outcome"] = display_df["predicted_outcome"].apply(
                lambda x: (x or "unknown").replace("_", " ").title()
            )
            display_df["damages"] = display_df["damages"].apply(
                lambda x: fmt_dollar(x) if x is not None else ""
            )
            display_df["attorney_fees"] = display_df["attorney_fees"].apply(
                lambda x: fmt_dollar(x) if x is not None else ""
            )
            display_df["similarity_score"] = display_df["similarity_score"].apply(
                lambda x: f"{x:.3f}"
            )

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "title": st.column_config.TextColumn("Case Title", width="large"),
                    "circuit": st.column_config.TextColumn("Circuit"),
                    "date_issued": st.column_config.TextColumn("Date"),
                    "predicted_outcome": st.column_config.TextColumn("Outcome"),
                    "damages": st.column_config.TextColumn("Damages"),
                    "attorney_fees": st.column_config.TextColumn("Atty Fees"),
                    "similarity_score": st.column_config.TextColumn("Similarity"),
                },
            )

    # =====================================================================
    # Tab 2 — Risk Factors
    # =====================================================================
    with tab_risk:
        st.subheader("Defense Risk Factors")

        if not result["risk_factors"]:
            st.info("No defense risk factors found in similar cases.")
        else:
            df_risk = pd.DataFrame(result["risk_factors"])

            # Display table
            display_risk = df_risk.copy()
            display_risk["loss_rate"] = display_risk["loss_rate"].apply(fmt_pct)
            st.dataframe(
                display_risk,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "defense_type": st.column_config.TextColumn("Defense Type"),
                    "count": st.column_config.NumberColumn("Times Raised"),
                    "loss_count": st.column_config.NumberColumn("Losses"),
                    "loss_rate": st.column_config.TextColumn("Loss Rate"),
                },
            )

            # Bar chart of defense frequency
            fig_def = px.bar(
                df_risk.sort_values("count", ascending=True),
                x="count", y="defense_type",
                orientation="h",
                labels={"count": "Frequency", "defense_type": "Defense Type"},
                title="Defense Frequency in Similar Cases",
            )
            fig_def.update_layout(yaxis_title="", height=max(300, len(df_risk) * 35))
            st.plotly_chart(fig_def, use_container_width=True)

    # =====================================================================
    # Tab 3 — Claim Recommendations
    # =====================================================================
    with tab_claims:
        st.subheader("Claim Section Recommendations")

        if not result["claim_recommendations"]:
            st.info("No claim recommendations available from similar cases.")
        else:
            df_claims = pd.DataFrame(result["claim_recommendations"])

            display_claims = df_claims.copy()
            display_claims["claim_section"] = display_claims["claim_section"].apply(
                lambda x: f"\u00a7{x}" if not str(x).startswith("\u00a7") else x
            )
            display_claims["plaintiff_win_rate"] = display_claims["plaintiff_win_rate"].apply(fmt_pct)
            display_claims["avg_damages"] = display_claims["avg_damages"].apply(
                lambda x: fmt_dollar(x) if x is not None else ""
            )

            st.dataframe(
                display_claims,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "claim_section": st.column_config.TextColumn("Claim Section"),
                    "count": st.column_config.NumberColumn("Count"),
                    "plaintiff_win_rate": st.column_config.TextColumn("Plaintiff Win Rate"),
                    "avg_damages": st.column_config.TextColumn("Avg Damages"),
                },
            )
