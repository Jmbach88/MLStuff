import streamlit as st
from sqlalchemy import text

from db import get_local_engine, init_local_db, get_session

st.set_page_config(page_title="Analytics", page_icon="⚖️", layout="wide")
st.title("Trends &amp; Analytics")


@st.cache_resource
def get_db_engine():
    engine = get_local_engine()
    init_local_db(engine)
    return engine


engine = get_db_engine()
session = get_session(engine)

# Check if predictions exist
pred_count = session.execute(text("SELECT COUNT(*) FROM predictions WHERE label_type='outcome'")).fetchone()[0]

if pred_count == 0:
    st.warning("No predictions found. Run `python classify.py` to train models and generate predictions.")
    st.stop()

# --- Sidebar Filters ---
with st.sidebar:
    st.header("Filters")

    statute_options = ["All", "FDCPA", "TCPA", "FCRA"]
    selected_statute = st.selectbox("Statute", statute_options)

    circuits = sorted([
        r[0] for r in session.execute(
            text("SELECT DISTINCT circuit FROM opinions WHERE circuit IS NOT NULL AND circuit != ''")
        ).fetchall()
    ])
    selected_circuit = st.selectbox("Circuit", ["All"] + circuits)

    court_types = ["All", "district", "circuit"]
    selected_court_type = st.selectbox("Court Type", court_types)

# --- Build filter SQL clause ---
filter_clauses = []
if selected_statute != "All":
    filter_clauses.append(
        f"o.id IN (SELECT os.opinion_id FROM opinion_statutes os "
        f"JOIN statutes s ON os.statute_id = s.id WHERE UPPER(s.key) = '{selected_statute.upper()}')"
    )
if selected_circuit != "All":
    filter_clauses.append(f"o.circuit = '{selected_circuit}'")
if selected_court_type != "All":
    filter_clauses.append(f"o.court_type = '{selected_court_type}'")

where_clause = " AND ".join(filter_clauses) if filter_clauses else "1=1"

# --- Outcome Distribution ---
st.subheader("Outcome Distribution")

outcome_data = session.execute(text(
    f"SELECT p.predicted_value, COUNT(*) "
    f"FROM predictions p JOIN opinions o ON p.opinion_id = o.id "
    f"WHERE p.label_type = 'outcome' AND p.model_name = 'outcome_logreg_v1' "
    f"AND {where_clause} "
    f"GROUP BY p.predicted_value ORDER BY COUNT(*) DESC"
)).fetchall()

if outcome_data:
    col1, col2 = st.columns([2, 1])
    with col1:
        chart_data = {r[0]: r[1] for r in outcome_data}
        st.bar_chart(chart_data)
    with col2:
        total = sum(r[1] for r in outcome_data)
        for label, count in outcome_data:
            pct = count / total * 100
            display_label = label.replace("_", " ").title()
            st.metric(display_label, f"{count:,}", f"{pct:.1f}%")

# --- Outcome by Statute ---
st.subheader("Outcome by Statute")

statute_outcome = session.execute(text(
    "SELECT UPPER(s.key), p.predicted_value, COUNT(*) "
    "FROM predictions p "
    "JOIN opinions o ON p.opinion_id = o.id "
    "JOIN opinion_statutes os ON o.id = os.opinion_id "
    "JOIN statutes s ON os.statute_id = s.id "
    "WHERE p.label_type = 'outcome' AND p.model_name = 'outcome_logreg_v1' "
    "GROUP BY UPPER(s.key), p.predicted_value "
    "ORDER BY UPPER(s.key), p.predicted_value"
)).fetchall()

if statute_outcome:
    import pandas as pd
    df = pd.DataFrame(statute_outcome, columns=["Statute", "Outcome", "Count"])
    pivot = df.pivot_table(index="Statute", columns="Outcome", values="Count", fill_value=0)
    st.bar_chart(pivot)

# --- Outcome by Circuit ---
st.subheader("Outcome by Circuit")

circuit_outcome = session.execute(text(
    f"SELECT o.circuit, p.predicted_value, COUNT(*) "
    f"FROM predictions p JOIN opinions o ON p.opinion_id = o.id "
    f"WHERE p.label_type = 'outcome' AND p.model_name = 'outcome_logreg_v1' "
    f"AND o.circuit IS NOT NULL AND o.circuit != '' "
    f"AND {where_clause} "
    f"GROUP BY o.circuit, p.predicted_value "
    f"ORDER BY o.circuit"
)).fetchall()

if circuit_outcome:
    import pandas as pd
    df = pd.DataFrame(circuit_outcome, columns=["Circuit", "Outcome", "Count"])
    pivot = df.pivot_table(index="Circuit", columns="Outcome", values="Count", fill_value=0)
    st.bar_chart(pivot)

# --- Top Claim Sections ---
st.subheader("Top Statutory Sections Cited")

claim_data = session.execute(text(
    f"SELECT p.predicted_value, COUNT(*) "
    f"FROM predictions p JOIN opinions o ON p.opinion_id = o.id "
    f"WHERE p.label_type = 'claim_type' AND p.model_name = 'claim_type_logreg_v1' "
    f"AND {where_clause} "
    f"GROUP BY p.predicted_value ORDER BY COUNT(*) DESC LIMIT 20"
)).fetchall()

if claim_data:
    chart_data = {f"§{r[0]}": r[1] for r in claim_data}
    st.bar_chart(chart_data)

# --- Model Performance ---
st.subheader("Model Performance")

models = session.execute(text(
    "SELECT name, label_type, accuracy, f1_score, trained_at FROM models ORDER BY trained_at DESC"
)).fetchall()

if models:
    for name, ltype, acc, f1, trained in models:
        with st.expander(f"{name} ({ltype})"):
            col1, col2, col3 = st.columns(3)
            with col1:
                if acc is not None:
                    st.metric("Accuracy", f"{acc:.3f}")
            with col2:
                if f1 is not None:
                    st.metric("Macro F1", f"{f1:.3f}")
            with col3:
                st.metric("Trained", trained[:10] if trained else "N/A")

# --- Summary Stats ---
st.subheader("Dataset Summary")

total_opinions = session.execute(text("SELECT COUNT(*) FROM opinions")).fetchone()[0]
labeled_outcomes = session.execute(text(
    "SELECT COUNT(DISTINCT opinion_id) FROM labels WHERE label_type='outcome' AND label_value != 'unlabeled'"
)).fetchone()[0]
predicted_outcomes = session.execute(text(
    "SELECT COUNT(DISTINCT opinion_id) FROM predictions WHERE label_type='outcome'"
)).fetchone()[0]

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Opinions", f"{total_opinions:,}")
with col2:
    st.metric("Labeled (regex)", f"{labeled_outcomes:,}")
with col3:
    st.metric("Predicted", f"{predicted_outcomes:,}")

session.close()
