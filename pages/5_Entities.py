import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import text

from db import get_local_engine, init_local_db, get_session

st.set_page_config(page_title="Entities", page_icon="⚖️", layout="wide")
st.title("Named Entities")


@st.cache_resource
def get_db_engine():
    engine = get_local_engine()
    init_local_db(engine)
    return engine


@st.cache_data
def load_entity_summary(_engine):
    with _engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM entities")).scalar()
        types = conn.execute(text(
            "SELECT entity_type, COUNT(*) FROM entities GROUP BY entity_type ORDER BY COUNT(*) DESC"
        )).fetchall()
    return total or 0, types


@st.cache_data
def load_damages_by_circuit(_engine):
    session = get_session(_engine)
    rows = session.execute(text(
        "SELECT o.circuit, e.entity_value "
        "FROM entities e "
        "JOIN opinions o ON e.opinion_id = o.id "
        "WHERE e.entity_type = 'DAMAGES_AWARDED' "
        "AND o.circuit IS NOT NULL AND o.circuit != ''"
    )).fetchall()
    session.close()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["circuit", "amount_str"])
    df["amount"] = df["amount_str"].str.replace(r'[\$,]', '', regex=True).astype(float)
    stats = df.groupby("circuit")["amount"].agg(["mean", "median", "count"]).reset_index()
    stats.columns = ["circuit", "avg_damages", "median_damages", "count"]
    return stats.sort_values("avg_damages", ascending=False)


@st.cache_data
def load_fees_by_circuit(_engine):
    session = get_session(_engine)
    rows = session.execute(text(
        "SELECT o.circuit, e.entity_value "
        "FROM entities e "
        "JOIN opinions o ON e.opinion_id = o.id "
        "WHERE e.entity_type = 'ATTORNEY_FEES' "
        "AND o.circuit IS NOT NULL AND o.circuit != ''"
    )).fetchall()
    session.close()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["circuit", "amount_str"])
    df["amount"] = df["amount_str"].str.replace(r'[\$,]', '', regex=True).astype(float)
    stats = df.groupby("circuit")["amount"].agg(["mean", "median", "count"]).reset_index()
    stats.columns = ["circuit", "avg_fees", "median_fees", "count"]
    return stats.sort_values("avg_fees", ascending=False)


@st.cache_data
def load_top_defendants(_engine, limit=20):
    session = get_session(_engine)
    rows = session.execute(text(
        "SELECT entity_value, COUNT(*) as cnt "
        "FROM entities "
        "WHERE entity_type = 'DEFENDANT' "
        "GROUP BY entity_value "
        "ORDER BY cnt DESC "
        "LIMIT :limit"
    ), {"limit": limit}).fetchall()
    session.close()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=["defendant", "cases"])


@st.cache_data
def load_top_judges(_engine, limit=20):
    session = get_session(_engine)
    rows = session.execute(text(
        "SELECT entity_value, COUNT(*) as cnt "
        "FROM entities "
        "WHERE entity_type = 'JUDGE' "
        "GROUP BY entity_value "
        "ORDER BY cnt DESC "
        "LIMIT :limit"
    ), {"limit": limit}).fetchall()
    session.close()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=["judge", "opinions"])


@st.cache_data
def load_debt_type_distribution(_engine):
    session = get_session(_engine)
    rows = session.execute(text(
        "SELECT entity_value, COUNT(*) as cnt "
        "FROM entities "
        "WHERE entity_type = 'DEBT_TYPE' "
        "GROUP BY entity_value "
        "ORDER BY cnt DESC"
    )).fetchall()
    session.close()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=["debt_type", "count"])


@st.cache_data
def load_entity_browser(_engine, entity_type=None, limit=200):
    session = get_session(_engine)
    query = (
        "SELECT e.entity_type, e.entity_value, o.title, o.circuit, o.date_issued, e.opinion_id "
        "FROM entities e "
        "JOIN opinions o ON e.opinion_id = o.id "
    )
    params = {"limit": limit}
    if entity_type and entity_type != "All":
        query += "WHERE e.entity_type = :etype "
        params["etype"] = entity_type
    query += "ORDER BY o.date_issued DESC LIMIT :limit"
    rows = session.execute(text(query), params).fetchall()
    session.close()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=[
        "entity_type", "entity_value", "title", "circuit", "date_issued", "opinion_id",
    ])


@st.cache_data
def load_opinion_entities(_engine, opinion_id):
    session = get_session(_engine)
    rows = session.execute(text(
        "SELECT entity_type, entity_value, context_snippet "
        "FROM entities WHERE opinion_id = :oid "
        "ORDER BY entity_type"
    ), {"oid": opinion_id}).fetchall()
    session.close()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=["entity_type", "entity_value", "context_snippet"])


engine = get_db_engine()
total_entities, type_counts = load_entity_summary(engine)

if total_entities == 0:
    st.warning("No entities found. Run `python ner.py` to extract entities.")
    st.stop()

# --- Sidebar ---
with st.sidebar:
    st.header("Filters")
    type_options = ["All"] + [r[0] for r in type_counts]
    selected_type = st.selectbox("Entity Type", type_options)

# --- Tab layout ---
tab_analytics, tab_browser = st.tabs(["Analytics", "Entity Browser"])

with tab_analytics:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Entities", f"{total_entities:,}")
    judges_count = next((r[1] for r in type_counts if r[0] == "JUDGE"), 0)
    damages_count = next((r[1] for r in type_counts if r[0] == "DAMAGES_AWARDED"), 0)
    fees_count = next((r[1] for r in type_counts if r[0] == "ATTORNEY_FEES"), 0)
    with col2:
        st.metric("Judges Found", f"{judges_count:,}")
    with col3:
        st.metric("Damages Awards", f"{damages_count:,}")
    with col4:
        st.metric("Attorney Fee Awards", f"{fees_count:,}")

    st.subheader("Damages Awarded by Circuit")
    df_damages = load_damages_by_circuit(engine)
    if not df_damages.empty:
        fig_dam = px.bar(
            df_damages, x="circuit", y="avg_damages",
            labels={"circuit": "Circuit", "avg_damages": "Average Damages ($)"},
            color="avg_damages", color_continuous_scale="Reds",
        )
        fig_dam.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig_dam, use_container_width=True)
    else:
        st.info("No damages data available.")

    st.subheader("Attorney Fees by Circuit")
    df_fees = load_fees_by_circuit(engine)
    if not df_fees.empty:
        fig_fees = px.bar(
            df_fees, x="circuit", y="avg_fees",
            labels={"circuit": "Circuit", "avg_fees": "Average Attorney Fees ($)"},
            color="avg_fees", color_continuous_scale="Blues",
        )
        fig_fees.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig_fees, use_container_width=True)

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Top Defendants")
        df_defendants = load_top_defendants(engine)
        if not df_defendants.empty:
            st.dataframe(df_defendants, use_container_width=True, hide_index=True)

    with col_right:
        st.subheader("Top Judges")
        df_judges = load_top_judges(engine)
        if not df_judges.empty:
            st.dataframe(df_judges, use_container_width=True, hide_index=True)

    st.subheader("Debt Type Distribution")
    df_debt = load_debt_type_distribution(engine)
    if not df_debt.empty:
        fig_debt = px.pie(df_debt, values="count", names="debt_type")
        fig_debt.update_layout(height=400)
        st.plotly_chart(fig_debt, use_container_width=True)

with tab_browser:
    st.subheader("Entity Browser")

    df_browse = load_entity_browser(engine, entity_type=selected_type, limit=200)
    if not df_browse.empty:
        st.dataframe(
            df_browse[["entity_type", "entity_value", "title", "circuit", "date_issued"]],
            use_container_width=True, hide_index=True,
        )

        st.subheader("Opinion Detail")
        opinion_ids = df_browse["opinion_id"].unique().tolist()
        if opinion_ids:
            selected_oid = st.selectbox("Select Opinion ID", opinion_ids)
            df_detail = load_opinion_entities(engine, selected_oid)
            if not df_detail.empty:
                st.dataframe(df_detail, use_container_width=True, hide_index=True)
    else:
        st.info("No entities match the current filter.")
