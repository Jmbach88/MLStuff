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
            text("SELECT COUNT(DISTINCT citing_opinion_id) FROM citations")
        ).scalar()
        unique_external = conn.execute(
            text("SELECT COUNT(*) FROM (SELECT DISTINCT volume, reporter, page FROM citations)")
        ).scalar()
    return {
        "total": total or 0,
        "resolved": resolved or 0,
        "opinions_citing": unique_cited or 0,
        "unique_external": unique_external or 0,
    }


@st.cache_data
def load_top_cited_cases(_engine, limit=30):
    """Most frequently cited cases across the corpus (by citation string)."""
    session = get_session(_engine)
    rows = session.execute(text(
        "SELECT citation_string, volume, reporter, page, COUNT(*) as times_cited "
        "FROM citations "
        "GROUP BY volume, reporter, page "
        "ORDER BY times_cited DESC "
        "LIMIT :limit"
    ), {"limit": limit}).fetchall()
    session.close()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=[
        "citation", "volume", "reporter", "page", "times_cited",
    ])


@st.cache_data
def load_reporter_distribution(_engine):
    """Citation counts by reporter type."""
    session = get_session(_engine)
    rows = session.execute(text(
        "SELECT reporter, COUNT(*) as count FROM citations "
        "GROUP BY reporter ORDER BY count DESC"
    )).fetchall()
    session.close()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=["reporter", "count"])


@st.cache_data
def load_most_citing_opinions(_engine, limit=20):
    """Opinions that cite the most other cases."""
    session = get_session(_engine)
    rows = session.execute(text(
        "SELECT c.citing_opinion_id, COUNT(*) as cite_count, "
        "o.title, o.court_type, o.circuit, o.date_issued "
        "FROM citations c "
        "JOIN opinions o ON c.citing_opinion_id = o.id "
        "GROUP BY c.citing_opinion_id "
        "ORDER BY cite_count DESC "
        "LIMIT :limit"
    ), {"limit": limit}).fetchall()
    session.close()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=[
        "opinion_id", "citations_made", "title", "court_type", "circuit", "date_issued",
    ])


@st.cache_data
def load_opinion_citations(_engine, opinion_id):
    """Load outgoing citations for one opinion."""
    session = get_session(_engine)
    outgoing = session.execute(text(
        "SELECT c.citation_string, c.reporter, c.context_snippet "
        "FROM citations c "
        "WHERE c.citing_opinion_id = :oid "
        "ORDER BY c.reporter, c.volume"
    ), {"oid": opinion_id}).fetchall()
    session.close()
    return outgoing


@st.cache_data
def load_opinion_statutes(_engine):
    """Map opinion_id -> statute key for filtering."""
    session = get_session(_engine)
    rows = session.execute(text(
        "SELECT os.opinion_id, UPPER(s.key) "
        "FROM opinion_statutes os "
        "JOIN statutes s ON os.statute_id = s.id"
    )).fetchall()
    session.close()
    return pd.DataFrame(rows, columns=["opinion_id", "statute"])


@st.cache_data
def load_opinions_for_select(_engine):
    """Load opinion IDs and titles ordered by citation count."""
    session = get_session(_engine)
    rows = session.execute(text(
        "SELECT c.citing_opinion_id, COUNT(*) as cnt, o.title "
        "FROM citations c "
        "JOIN opinions o ON c.citing_opinion_id = o.id "
        "GROUP BY c.citing_opinion_id "
        "ORDER BY cnt DESC"
    )).fetchall()
    session.close()
    return rows


@st.cache_data
def load_most_cited_internal(_engine, limit=20):
    """Top cases by in-degree (only if resolution found matches)."""
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
    st.metric("Opinions Citing", f"{summary['opinions_citing']:,}")
with col3:
    st.metric("Unique Cases Cited", f"{summary['unique_external']:,}")
with col4:
    avg = summary["total"] / summary["opinions_citing"] if summary["opinions_citing"] > 0 else 0
    st.metric("Avg Citations/Opinion", f"{avg:.1f}")

# --- Sidebar Filters ---
with st.sidebar:
    st.header("Filters")
    reporter_options = ["All"] + load_reporter_distribution(engine)["reporter"].tolist()
    selected_reporter = st.selectbox("Reporter", reporter_options)

# --- Most-Cited Cases ---
st.subheader("Most-Cited Cases")
df_top = load_top_cited_cases(engine, limit=30)
if not df_top.empty:
    if selected_reporter != "All":
        df_top = df_top[df_top["reporter"] == selected_reporter]
    st.dataframe(
        df_top[["citation", "reporter", "times_cited"]].head(20),
        use_container_width=True, hide_index=True,
    )

# --- Citations by Reporter ---
st.subheader("Citations by Reporter")
df_reporters = load_reporter_distribution(engine)
if not df_reporters.empty:
    fig_rep = px.bar(
        df_reporters, x="reporter", y="count",
        labels={"reporter": "Reporter", "count": "Citations"},
        color="count", color_continuous_scale="Blues",
    )
    fig_rep.update_layout(showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig_rep, use_container_width=True)

# --- Most-Citing Opinions ---
st.subheader("Most-Citing Opinions")
df_citing = load_most_citing_opinions(engine, limit=20)
if not df_citing.empty:
    st.dataframe(
        df_citing[["title", "court_type", "circuit", "date_issued", "citations_made"]],
        use_container_width=True, hide_index=True,
    )

# --- Internal Graph Metrics (if resolution found matches) ---
if summary["resolved"] > 0:
    st.subheader("Most-Cited Corpus Opinions (Internal)")
    df_internal = load_most_cited_internal(engine, limit=20)
    if not df_internal.empty:
        df_display = df_internal[["title", "court_type", "circuit", "date_issued", "in_degree", "pagerank"]].copy()
        df_display["pagerank"] = df_display["pagerank"].apply(lambda x: f"{x:.6f}")
        st.dataframe(df_display, use_container_width=True, hide_index=True)

# --- Per-Opinion Drill-Down ---
st.subheader("Per-Opinion Drill-Down")

opinions_list = load_opinions_for_select(engine)
if opinions_list:
    options = {f"{row[0]} — {row[2][:80]} ({row[1]} cites)": row[0] for row in opinions_list}
    selected_label = st.selectbox("Select an opinion", list(options.keys()))
    selected_id = options[selected_label]

    outgoing = load_opinion_citations(engine, selected_id)

    st.metric("Citations Made", len(outgoing))

    if outgoing:
        df_out = pd.DataFrame(outgoing, columns=[
            "citation_string", "reporter", "context_snippet",
        ])
        st.dataframe(
            df_out[["citation_string", "reporter"]],
            use_container_width=True, hide_index=True,
        )
