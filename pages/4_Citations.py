import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import text

from db import get_local_engine, init_local_db, get_session

st.set_page_config(page_title="Citations", page_icon="⚖️", layout="wide")
st.title("Citation Network")


@st.cache_resource
def get_db_engine():
    engine = get_local_engine()
    init_local_db(engine)
    return engine


@st.cache_data
def load_citation_summary(_engine):
    with _engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM citations")).scalar()
        resolved = conn.execute(
            text("SELECT COUNT(*) FROM citations WHERE cited_opinion_id IS NOT NULL")
        ).scalar()
        unique_cited = conn.execute(
            text("SELECT COUNT(DISTINCT cited_opinion_id) FROM citations "
                 "WHERE cited_opinion_id IS NOT NULL")
        ).scalar()
        n_metrics = conn.execute(
            text("SELECT COUNT(*) FROM opinion_metrics")
        ).scalar()
    return {
        "total": total or 0,
        "resolved": resolved or 0,
        "unique_cited": unique_cited or 0,
        "n_metrics": n_metrics or 0,
    }


@st.cache_data
def load_most_cited(_engine, limit=20):
    session = get_session(_engine)
    rows = session.execute(text(
        "SELECT m.opinion_id, m.in_degree, m.pagerank, m.community_id, "
        "o.title, o.court_type, o.circuit, o.date_issued "
        "FROM opinion_metrics m "
        "JOIN opinions o ON m.opinion_id = o.id "
        "ORDER BY m.in_degree DESC "
        "LIMIT :limit"
    ), {"limit": limit}).fetchall()
    session.close()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=[
        "opinion_id", "in_degree", "pagerank", "community_id",
        "title", "court_type", "circuit", "date_issued",
    ])


@st.cache_data
def load_most_influential(_engine, limit=20):
    session = get_session(_engine)
    rows = session.execute(text(
        "SELECT m.opinion_id, m.pagerank, m.in_degree, m.hub_score, "
        "m.authority_score, m.community_id, "
        "o.title, o.court_type, o.circuit, o.date_issued "
        "FROM opinion_metrics m "
        "JOIN opinions o ON m.opinion_id = o.id "
        "ORDER BY m.pagerank DESC "
        "LIMIT :limit"
    ), {"limit": limit}).fetchall()
    session.close()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=[
        "opinion_id", "pagerank", "in_degree", "hub_score",
        "authority_score", "community_id",
        "title", "court_type", "circuit", "date_issued",
    ])


@st.cache_data
def load_community_sizes(_engine):
    session = get_session(_engine)
    rows = session.execute(text(
        "SELECT community_id, COUNT(*) as size "
        "FROM opinion_metrics "
        "WHERE community_id IS NOT NULL "
        "GROUP BY community_id "
        "ORDER BY size DESC"
    )).fetchall()
    session.close()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=["community_id", "size"])


@st.cache_data
def load_opinion_citations(_engine, opinion_id):
    session = get_session(_engine)

    outgoing = session.execute(text(
        "SELECT c.citation_string, c.cited_opinion_id, c.context_snippet, "
        "o.title as cited_title "
        "FROM citations c "
        "LEFT JOIN opinions o ON c.cited_opinion_id = o.id "
        "WHERE c.citing_opinion_id = :oid"
    ), {"oid": opinion_id}).fetchall()

    incoming = session.execute(text(
        "SELECT o.title, o.id as citing_id "
        "FROM citations c "
        "JOIN opinions o ON c.citing_opinion_id = o.id "
        "WHERE c.cited_opinion_id = :oid"
    ), {"oid": opinion_id}).fetchall()

    metric = session.execute(text(
        "SELECT in_degree, out_degree, pagerank, hub_score, "
        "authority_score, community_id "
        "FROM opinion_metrics WHERE opinion_id = :oid"
    ), {"oid": opinion_id}).fetchone()

    session.close()
    return outgoing, incoming, metric


@st.cache_data
def load_opinions_for_select(_engine):
    session = get_session(_engine)
    rows = session.execute(text(
        "SELECT o.id, o.title FROM opinions o "
        "JOIN opinion_metrics m ON o.id = m.opinion_id "
        "ORDER BY m.in_degree DESC"
    )).fetchall()
    session.close()
    return rows


engine = get_db_engine()
summary = load_citation_summary(engine)

if summary["total"] == 0:
    st.warning("No citations found. Run `python citations.py` to extract citations.")
    st.stop()

# --- Summary Metrics ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Citations", f"{summary['total']:,}")
with col2:
    st.metric("Resolved (Internal)", f"{summary['resolved']:,}")
with col3:
    st.metric("Unique Cases Cited", f"{summary['unique_cited']:,}")
with col4:
    rate = (
        f"{summary['resolved'] / summary['total'] * 100:.1f}%"
        if summary["total"] > 0 else "N/A"
    )
    st.metric("Resolution Rate", rate)

# --- Most-Cited Cases ---
st.subheader("Most-Cited Cases (In-Degree)")
df_cited = load_most_cited(engine, limit=20)
if not df_cited.empty:
    display_cols = ["title", "court_type", "circuit", "date_issued", "in_degree", "pagerank"]
    df_display = df_cited[display_cols].copy()
    df_display["pagerank"] = df_display["pagerank"].apply(lambda x: f"{x:.6f}")
    st.dataframe(df_display, use_container_width=True, hide_index=True)
else:
    st.info("No metrics computed yet.")

# --- Most Influential (PageRank) ---
st.subheader("Most Influential Cases (PageRank)")
df_influential = load_most_influential(engine, limit=20)
if not df_influential.empty:
    display_cols = ["title", "court_type", "circuit", "pagerank", "in_degree", "authority_score"]
    df_display = df_influential[display_cols].copy()
    df_display["pagerank"] = df_display["pagerank"].apply(lambda x: f"{x:.6f}")
    df_display["authority_score"] = df_display["authority_score"].apply(lambda x: f"{x:.6f}")
    st.dataframe(df_display, use_container_width=True, hide_index=True)

# --- Citation Communities ---
st.subheader("Citation Communities")
df_communities = load_community_sizes(engine)
if not df_communities.empty:
    top_communities = df_communities.head(20)
    top_communities = top_communities.copy()
    top_communities["label"] = top_communities["community_id"].apply(
        lambda c: f"Community {c}"
    )
    fig_comm = px.bar(
        top_communities, x="label", y="size",
        labels={"label": "Community", "size": "Number of Opinions"},
        color="size", color_continuous_scale="Blues",
    )
    fig_comm.update_layout(showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig_comm, use_container_width=True)

# --- Per-Opinion Drill-Down ---
st.subheader("Per-Opinion Drill-Down")

opinions_list = load_opinions_for_select(engine)
if opinions_list:
    options = {f"{row[0]} — {row[1][:80]}": row[0] for row in opinions_list}
    selected_label = st.selectbox("Select an opinion", list(options.keys()))
    selected_id = options[selected_label]

    outgoing, incoming, metric = load_opinion_citations(engine, selected_id)

    if metric:
        mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
        with mcol1:
            st.metric("In-Degree", metric[0])
        with mcol2:
            st.metric("Out-Degree", metric[1])
        with mcol3:
            st.metric("PageRank", f"{metric[2]:.6f}")
        with mcol4:
            st.metric("Hub Score", f"{metric[3]:.6f}")
        with mcol5:
            st.metric("Authority", f"{metric[4]:.6f}")

    col_out, col_in = st.columns(2)

    with col_out:
        st.markdown("**Cites (Outgoing)**")
        if outgoing:
            df_out = pd.DataFrame(outgoing, columns=[
                "citation_string", "cited_opinion_id", "context_snippet", "cited_title",
            ])
            df_out["cited_title"] = df_out["cited_title"].fillna("(external)")
            st.dataframe(
                df_out[["citation_string", "cited_title"]],
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("No outgoing citations.")

    with col_in:
        st.markdown("**Cited By (Incoming)**")
        if incoming:
            df_in = pd.DataFrame(incoming, columns=["title", "citing_id"])
            st.dataframe(df_in, use_container_width=True, hide_index=True)
        else:
            st.info("No incoming citations from corpus.")
