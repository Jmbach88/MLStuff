import streamlit as st
from sqlalchemy import text

from db import get_local_engine, get_session, init_local_db
from index import load_index
from search import search_opinions


@st.cache_resource
def get_search_index():
    """Load FAISS index and chunk map, cached across reruns."""
    return load_index()


@st.cache_resource
def get_db_engine():
    engine = get_local_engine()
    init_local_db(engine)
    return engine


def get_filter_options(engine):
    """Load filter options from the local DB."""
    session = get_session(engine)
    circuits = sorted([
        r[0] for r in session.execute(
            text("SELECT DISTINCT circuit FROM opinions WHERE circuit IS NOT NULL AND circuit != ''")
        ).fetchall()
    ])
    subsections = sorted([
        r[0] for r in session.execute(
            text("SELECT DISTINCT subsection FROM fdcpa_sections ORDER BY subsection")
        ).fetchall()
    ])
    session.close()
    return circuits, subsections


def get_full_opinion_text(engine, opinion_id: int) -> str:
    """Retrieve the full opinion text from the local DB."""
    session = get_session(engine)
    row = session.execute(
        text("SELECT plain_text FROM opinions WHERE id = :oid"),
        {"oid": opinion_id},
    ).fetchone()
    session.close()
    return row[0] if row else ""


def get_opinion_ids_for_subsections(engine, subsections: list[str]) -> list[int]:
    """Get opinion IDs that cite any of the given FDCPA subsections."""
    session = get_session(engine)
    placeholders = ",".join(f"'{s}'" for s in subsections)
    rows = session.execute(text(
        f"SELECT DISTINCT opinion_id FROM fdcpa_sections WHERE subsection IN ({placeholders})"
    )).fetchall()
    session.close()
    return [r[0] for r in rows]


# --- Page Config ---
st.set_page_config(page_title="Search", page_icon="⚖️", layout="wide")
st.title("Semantic Search")

engine = get_db_engine()
index, chunk_map = get_search_index()

# --- Sidebar Filters ---
with st.sidebar:
    st.header("Filters")

    statute_options = ["FDCPA", "TCPA", "FCRA"]
    selected_statutes = st.multiselect("Statute", statute_options)

    circuits, subsections = get_filter_options(engine)
    selected_circuits = st.multiselect("Circuit", circuits)

    court_type = st.radio("Court Type", ["All", "District", "Circuit"], horizontal=True)

    col1, col2 = st.columns(2)
    with col1:
        date_from = st.text_input("Date from", placeholder="YYYY-MM-DD")
    with col2:
        date_to = st.text_input("Date to", placeholder="YYYY-MM-DD")

    # FDCPA subsection filter
    selected_subsections = []
    if subsections and ("FDCPA" in selected_statutes or not selected_statutes):
        selected_subsections = st.multiselect("FDCPA Subsections", subsections)

    results_per_page = st.slider("Results per page", 5, 50, 10)

# --- Search Box ---
query = st.text_input(
    "Search opinions",
    placeholder="e.g., collector called third party and disclosed debt",
    label_visibility="collapsed",
)

# --- Execute Search ---
if index is None:
    st.warning("No index found. Run `python pipeline.py` to process opinions first.")
elif query:
    # Build filters
    filters = {}

    if selected_statutes and len(selected_statutes) == 1:
        filters["statute"] = selected_statutes[0]

    if selected_circuits and len(selected_circuits) == 1:
        filters["circuit"] = selected_circuits[0]

    if court_type != "All":
        filters["court_type"] = court_type.lower()

    if date_from:
        filters["date_from"] = date_from
    if date_to:
        filters["date_to"] = date_to

    # Handle FDCPA subsection filter via SQLite lookup
    if selected_subsections:
        opinion_ids = get_opinion_ids_for_subsections(engine, selected_subsections)
        if opinion_ids:
            filters["opinion_ids"] = opinion_ids
        else:
            st.warning("No opinions match the selected subsections.")
            st.stop()

    # Handle multi-select filters client-side
    multi_statute_filter = selected_statutes if len(selected_statutes) > 1 else None
    multi_circuit_filter = selected_circuits if len(selected_circuits) > 1 else None

    effective_top_k = results_per_page
    if multi_statute_filter or multi_circuit_filter:
        effective_top_k = results_per_page * 3

    with st.spinner("Searching..."):
        results = search_opinions(index, chunk_map, query, top_k=effective_top_k, filters=filters)

    # Client-side filtering for multi-select
    if multi_statute_filter:
        results = [r for r in results if any(s in r["statutes"] for s in multi_statute_filter)]
    if multi_circuit_filter:
        results = [r for r in results if r["circuit"] in multi_circuit_filter]

    results = results[:results_per_page]

    # --- Display Results ---
    if not results:
        st.info("No results found. Try broadening your search or adjusting filters.")
    else:
        st.markdown(f"**{len(results)} results**")

        for i, r in enumerate(results):
            score_pct = r["similarity_score"] * 100

            st.markdown(f"### {r['title']}")

            meta_parts = []
            if r["court_name"]:
                meta_parts.append(r["court_name"])
            if r["circuit"]:
                meta_parts.append(f"{r['circuit']} Circuit")
            if r["date_issued"]:
                meta_parts.append(r["date_issued"])
            st.caption(" | ".join(meta_parts))

            if r["statutes"]:
                statute_list = r["statutes"].split(",")
                st.markdown(" ".join(f"`{s}`" for s in statute_list))

            st.progress(min(score_pct / 100, 1.0), text=f"Relevance: {score_pct:.1f}%")

            # Best matching passage
            passage = r.get("best_passage", "")
            if passage:
                display_passage = passage[:500] + ("..." if len(passage) > 500 else "")
                st.markdown(f"> {display_passage}")

            # Expandable full text
            with st.expander("Show full opinion"):
                full_text = get_full_opinion_text(engine, r["opinion_id"])
                if full_text:
                    st.text(full_text)
                else:
                    st.warning("Full text not available.")

            st.divider()
else:
    if index is not None:
        st.info(f"Ready to search {index.ntotal:,} chunks from the federal opinion corpus.")
