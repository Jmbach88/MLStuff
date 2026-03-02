import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import text

from db import get_local_engine, init_local_db
from trends import (
    outcome_trends_by_year,
    damages_trends_by_year,
    judge_stats,
    circuit_comparison,
    circuit_outcome_heatmap,
    defense_frequency,
    defense_effectiveness,
    defense_by_circuit,
    compare_outcome_rates,
)

st.set_page_config(page_title="Trends", page_icon="⚖️", layout="wide")
st.title("Trends & Analytics")


@st.cache_resource
def get_db_engine():
    engine = get_local_engine()
    init_local_db(engine)
    return engine


engine = get_db_engine()

# --- Check data exists ---
with engine.connect() as conn:
    pred_count = conn.execute(
        text("SELECT COUNT(*) FROM predictions WHERE label_type='outcome'")
    ).scalar()

if pred_count == 0:
    st.warning("No predictions found. Run `python classify.py` first.")
    st.stop()

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

    with engine.connect() as conn:
        year_range = conn.execute(text(
            "SELECT MIN(CAST(SUBSTR(date_issued, 1, 4) AS INTEGER)), "
            "       MAX(CAST(SUBSTR(date_issued, 1, 4) AS INTEGER)) "
            "FROM opinions "
            "WHERE date_issued IS NOT NULL AND date_issued != '' "
            "AND LENGTH(date_issued) >= 4"
        )).fetchone()
    db_year_min = year_range[0] or 1990
    db_year_max = year_range[1] or 2026
    year_min, year_max = st.slider(
        "Date Range",
        min_value=db_year_min, max_value=db_year_max,
        value=(db_year_min, db_year_max),
    )

# Resolve filter args
f_statute = selected_statute if selected_statute != "All" else None
f_circuit = selected_circuit if selected_circuit != "All" else None
f_court = selected_court_type if selected_court_type != "All" else None

# --- Helper formatters ---

def fmt_pct(v):
    if v is None or pd.isna(v):
        return ""
    return f"{v * 100:.1f}%"


def fmt_dollar(v):
    if v is None or pd.isna(v):
        return ""
    return f"${v:,.0f}"


# --- Tabs ---
tab_outcome, tab_damages, tab_judges, tab_circuits, tab_defense = st.tabs([
    "Outcome Trends", "Damages Trends", "Judicial Analytics",
    "Circuit Comparison", "Defense Analysis",
])

# =====================================================================
# Tab 1 — Outcome Trends
# =====================================================================
with tab_outcome:
    st.subheader("Plaintiff Win Rate Over Time")

    df_ot = outcome_trends_by_year(
        engine, statute=f_statute, circuit=f_circuit,
        court_type=f_court, year_min=year_min, year_max=year_max,
    )

    if df_ot.empty:
        st.info("No outcome data for the selected filters.")
    else:
        # If a specific circuit is selected, also get overall for comparison
        if f_circuit:
            df_overall = outcome_trends_by_year(
                engine, statute=f_statute, circuit=None,
                court_type=f_court, year_min=year_min, year_max=year_max,
            )
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_ot["year"], y=df_ot["plaintiff_win_rate"],
                mode="lines+markers", name=f_circuit,
            ))
            fig.add_trace(go.Scatter(
                x=df_overall["year"], y=df_overall["plaintiff_win_rate"],
                mode="lines+markers", name="Overall",
                line=dict(dash="dash"),
            ))
            fig.update_layout(
                yaxis_title="Plaintiff Win Rate",
                xaxis_title="Year",
                yaxis_tickformat=".0%",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Significance badge comparing circuit to overall
            circuit_wins = int(df_ot["plaintiff_wins"].sum())
            circuit_total = int(df_ot["total"].sum())
            overall_wins = int(df_overall["plaintiff_wins"].sum())
            overall_total = int(df_overall["total"].sum())
            if circuit_total > 0 and overall_total > 0:
                sig = compare_outcome_rates(
                    circuit_wins / circuit_total, circuit_total,
                    overall_wins / overall_total, overall_total,
                )
                circuit_rate = fmt_pct(circuit_wins / circuit_total)
                overall_rate = fmt_pct(overall_wins / overall_total)
                badge = sig["badge"] if sig["badge"] else "n.s."
                st.markdown(
                    f"**{f_circuit}** win rate {circuit_rate} vs overall "
                    f"{overall_rate} — significance: **{badge}** "
                    f"(p={sig['p_value']:.4f})"
                )
        else:
            fig = px.line(
                df_ot, x="year", y="plaintiff_win_rate",
                markers=True,
                labels={"plaintiff_win_rate": "Plaintiff Win Rate", "year": "Year"},
            )
            fig.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

# =====================================================================
# Tab 2 — Damages Trends
# =====================================================================
with tab_damages:
    st.subheader("Damages & Fees Over Time")

    df_dmg = damages_trends_by_year(
        engine, statute=f_statute, circuit=f_circuit,
        court_type=f_court, year_min=year_min, year_max=year_max,
    )

    if df_dmg.empty:
        st.info("No damages data for the selected filters.")
    else:
        col_d, col_f = st.columns(2)

        with col_d:
            st.markdown("**Median Damages by Year**")
            fig_d = px.line(
                df_dmg, x="year", y="median_damages",
                markers=True,
                labels={"median_damages": "Median Damages ($)", "year": "Year"},
            )
            st.plotly_chart(fig_d, use_container_width=True)

        with col_f:
            st.markdown("**Median Attorney Fees by Year**")
            fig_f = px.line(
                df_dmg, x="year", y="median_fees",
                markers=True,
                labels={"median_fees": "Median Fees ($)", "year": "Year"},
            )
            st.plotly_chart(fig_f, use_container_width=True)

        # Data table
        st.markdown("**Summary Table**")
        display_dmg = df_dmg.copy()
        display_dmg["median_damages"] = display_dmg["median_damages"].apply(fmt_dollar)
        display_dmg["mean_damages"] = display_dmg["mean_damages"].apply(fmt_dollar)
        display_dmg["median_fees"] = display_dmg["median_fees"].apply(fmt_dollar)
        display_dmg["mean_fees"] = display_dmg["mean_fees"].apply(fmt_dollar)
        st.dataframe(display_dmg, use_container_width=True, hide_index=True)

# =====================================================================
# Tab 3 — Judicial Analytics
# =====================================================================
with tab_judges:
    st.subheader("Judicial Analytics (10+ opinions)")

    df_judges = judge_stats(engine, min_opinions=10)

    if df_judges.empty:
        st.info("No judge data with 10+ opinions.")
    else:
        # Compute corpus-wide plaintiff win rate for significance comparison
        with engine.connect() as conn:
            corpus_row = conn.execute(text(
                "SELECT SUM(CASE WHEN predicted_value = 'plaintiff_win' THEN 1 ELSE 0 END), "
                "       COUNT(*) "
                "FROM predictions "
                "WHERE label_type = 'outcome' AND model_name = 'outcome_logreg_v1'"
            )).fetchone()
        corpus_wins = corpus_row[0] or 0
        corpus_total = corpus_row[1] or 1
        corpus_rate = corpus_wins / corpus_total

        # Add significance badge per judge
        badges = []
        for _, row in df_judges.iterrows():
            sig = compare_outcome_rates(
                row["plaintiff_win_rate"], int(row["opinion_count"]),
                corpus_rate, corpus_total,
            )
            badges.append(sig["badge"] if sig["badge"] else "")
        df_judges["sig"] = badges

        # Format for display
        display_j = df_judges.copy()
        display_j["plaintiff_win_rate"] = display_j["plaintiff_win_rate"].apply(fmt_pct)
        display_j["avg_damages"] = display_j["avg_damages"].apply(fmt_dollar)
        display_j["avg_fees"] = display_j["avg_fees"].apply(fmt_dollar)

        st.dataframe(
            display_j,
            use_container_width=True,
            hide_index=True,
            column_config={
                "judge": st.column_config.TextColumn("Judge"),
                "opinion_count": st.column_config.NumberColumn("Opinions"),
                "plaintiff_win_rate": st.column_config.TextColumn("Win Rate"),
                "avg_damages": st.column_config.TextColumn("Avg Damages"),
                "avg_fees": st.column_config.TextColumn("Avg Fees"),
                "sig": st.column_config.TextColumn("Sig", width="small"),
            },
        )
        st.caption("Significance vs corpus average: *** p<0.001, ** p<0.01, * p<0.05")

# =====================================================================
# Tab 4 — Circuit Comparison
# =====================================================================
with tab_circuits:
    st.subheader("Circuit Comparison")

    # --- Heatmap: plaintiff win rate by circuit x statute ---
    df_heat = circuit_outcome_heatmap(engine)
    if not df_heat.empty:
        pivot = df_heat.pivot_table(
            index="circuit", columns="statute",
            values="plaintiff_win_rate", aggfunc="first",
        )
        fig_heat = px.imshow(
            pivot,
            color_continuous_scale="RdYlGn",
            labels=dict(color="Win Rate"),
            text_auto=".0%",
            aspect="auto",
        )
        fig_heat.update_layout(
            title="Plaintiff Win Rate: Circuit x Statute",
            xaxis_title="Statute",
            yaxis_title="Circuit",
        )
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("No circuit-statute outcome data available.")

    # --- Bar charts: avg damages and fees by circuit ---
    df_circ = circuit_comparison(engine)
    if not df_circ.empty:
        col_cd, col_cf = st.columns(2)
        with col_cd:
            st.markdown("**Avg Damages by Circuit**")
            df_cd = df_circ.dropna(subset=["avg_damages"]).sort_values("avg_damages", ascending=False)
            if not df_cd.empty:
                fig_cd = px.bar(
                    df_cd, x="circuit", y="avg_damages",
                    labels={"avg_damages": "Avg Damages ($)", "circuit": "Circuit"},
                    color="avg_damages", color_continuous_scale="Reds",
                )
                fig_cd.update_layout(showlegend=False, coloraxis_showscale=False)
                st.plotly_chart(fig_cd, use_container_width=True)

        with col_cf:
            st.markdown("**Avg Attorney Fees by Circuit**")
            df_cf = df_circ.dropna(subset=["avg_fees"]).sort_values("avg_fees", ascending=False)
            if not df_cf.empty:
                fig_cf = px.bar(
                    df_cf, x="circuit", y="avg_fees",
                    labels={"avg_fees": "Avg Fees ($)", "circuit": "Circuit"},
                    color="avg_fees", color_continuous_scale="Blues",
                )
                fig_cf.update_layout(showlegend=False, coloraxis_showscale=False)
                st.plotly_chart(fig_cf, use_container_width=True)

        # Summary table with significance badges
        st.markdown("**Circuit Summary**")
        corpus_circ_total = int(df_circ["opinion_count"].sum())
        corpus_circ_rate = (
            (df_circ["plaintiff_win_rate"] * df_circ["opinion_count"]).sum()
            / corpus_circ_total
        ) if corpus_circ_total > 0 else 0

        circ_badges = []
        for _, row in df_circ.iterrows():
            sig = compare_outcome_rates(
                row["plaintiff_win_rate"], int(row["opinion_count"]),
                corpus_circ_rate, corpus_circ_total,
            )
            circ_badges.append(sig["badge"] if sig["badge"] else "")
        df_circ_display = df_circ.copy()
        df_circ_display["sig"] = circ_badges
        df_circ_display["plaintiff_win_rate"] = df_circ_display["plaintiff_win_rate"].apply(fmt_pct)
        df_circ_display["avg_damages"] = df_circ_display["avg_damages"].apply(fmt_dollar)
        df_circ_display["avg_fees"] = df_circ_display["avg_fees"].apply(fmt_dollar)

        st.dataframe(
            df_circ_display[["circuit", "opinion_count", "plaintiff_win_rate",
                             "avg_damages", "avg_fees", "sig"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "circuit": st.column_config.TextColumn("Circuit"),
                "opinion_count": st.column_config.NumberColumn("Opinions"),
                "plaintiff_win_rate": st.column_config.TextColumn("Win Rate"),
                "avg_damages": st.column_config.TextColumn("Avg Damages"),
                "avg_fees": st.column_config.TextColumn("Avg Fees"),
                "sig": st.column_config.TextColumn("Sig", width="small"),
            },
        )
        st.caption("Significance vs corpus average: *** p<0.001, ** p<0.01, * p<0.05")
    else:
        st.info("No circuit comparison data available.")

# =====================================================================
# Tab 5 — Defense Analysis
# =====================================================================
with tab_defense:
    st.subheader("Defense Analysis")

    # --- Horizontal bar: defense frequency top 20 ---
    df_freq = defense_frequency(engine)
    if not df_freq.empty:
        st.markdown("**Top 20 Defenses by Frequency**")
        df_top20 = df_freq.head(20).sort_values("count", ascending=True)
        fig_bar = px.bar(
            df_top20, x="count", y="defense_type",
            orientation="h",
            labels={"count": "Frequency", "defense_type": "Defense Type"},
        )
        fig_bar.update_layout(yaxis_title="", height=600)
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No defense frequency data available.")

    # --- Table: defense effectiveness with significance badges ---
    df_eff = defense_effectiveness(engine, min_count=20)
    if not df_eff.empty:
        st.markdown("**Defense Effectiveness**")
        eff_badges = []
        for _, row in df_eff.iterrows():
            sig = compare_outcome_rates(
                row["defendant_win_rate_when_raised"], int(row["times_raised"]),
                row["overall_defendant_win_rate"], pred_count,
            )
            eff_badges.append(sig["badge"] if sig["badge"] else "")

        df_eff_display = df_eff.copy()
        df_eff_display["sig"] = eff_badges
        df_eff_display["defendant_win_rate_when_raised"] = df_eff_display["defendant_win_rate_when_raised"].apply(fmt_pct)
        df_eff_display["overall_defendant_win_rate"] = df_eff_display["overall_defendant_win_rate"].apply(fmt_pct)

        st.dataframe(
            df_eff_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "defense_type": st.column_config.TextColumn("Defense"),
                "times_raised": st.column_config.NumberColumn("Times Raised"),
                "defendant_win_rate_when_raised": st.column_config.TextColumn("Def Win Rate (w/ defense)"),
                "overall_defendant_win_rate": st.column_config.TextColumn("Overall Def Win Rate"),
                "sig": st.column_config.TextColumn("Sig", width="small"),
            },
        )
        st.caption("Significance vs overall rate: *** p<0.001, ** p<0.01, * p<0.05")
    else:
        st.info("No defense effectiveness data (need 20+ occurrences).")

    # --- Heatmap: defense frequency by circuit (top 10) ---
    df_dbc = defense_by_circuit(engine)
    if not df_dbc.empty:
        st.markdown("**Defense Frequency by Circuit (Top 10 Defenses)**")
        top10_defenses = (
            df_dbc.groupby("defense_type")["count"].sum()
            .nlargest(10).index.tolist()
        )
        df_dbc_top = df_dbc[df_dbc["defense_type"].isin(top10_defenses)]
        pivot_dbc = df_dbc_top.pivot_table(
            index="defense_type", columns="circuit",
            values="count", aggfunc="sum", fill_value=0,
        )
        fig_dbc = px.imshow(
            pivot_dbc,
            color_continuous_scale="Blues",
            labels=dict(color="Count"),
            text_auto=True,
            aspect="auto",
        )
        fig_dbc.update_layout(
            xaxis_title="Circuit",
            yaxis_title="Defense Type",
            height=500,
        )
        st.plotly_chart(fig_dbc, use_container_width=True)
    else:
        st.info("No defense-by-circuit data available.")
