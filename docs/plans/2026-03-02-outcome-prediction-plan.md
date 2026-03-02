# Advanced Outcome Prediction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a case evaluation tool where an attorney inputs a case summary and gets predicted outcome, damages range, similar precedents, risk factors, and claim recommendations via similarity-weighted averaging over the FAISS index.

**Architecture:** Embed the input text with the same sentence-transformers model, search FAISS for top-50 similar opinions, load their outcome/entity/claim data from SQLite, compute weighted predictions using similarity scores as weights. No new DB tables, no model training.

**Tech Stack:** sentence-transformers, faiss-cpu, numpy, pandas, streamlit, plotly, sqlalchemy

---

### Task 1: Create predictor.py with core prediction logic

**Files:**
- Create: `predictor.py`
- Create: `tests/test_predictor.py`

**Step 1: Write failing tests**

Create `tests/test_predictor.py`:

```python
import os
import pytest
import numpy as np

os.environ["ML_LOCAL_DB"] = ":memory:"

from db import get_local_engine, init_local_db, get_session, Opinion, Entity
from sqlalchemy import text as sql_text


@pytest.fixture
def prediction_env():
    """Build in-memory DB + small FAISS index for testing predictions."""
    from embed import embed_chunks
    from index import build_index, add_to_index

    engine = get_local_engine()
    init_local_db(engine)
    session = get_session(engine)

    # Clean slate
    for table in ["entities", "predictions", "labels", "opinion_statutes",
                   "opinions", "statutes"]:
        session.execute(sql_text(f"DELETE FROM {table}"))
    session.commit()

    # Create statutes
    session.execute(sql_text(
        "INSERT INTO statutes (id, key, name) VALUES (1, 'fdcpa', 'FDCPA')"
    ))
    session.execute(sql_text(
        "INSERT INTO statutes (id, key, name) VALUES (2, 'tcpa', 'TCPA')"
    ))
    session.commit()

    # Create opinions with varied outcomes
    opinions_data = [
        (1, "pkg1", "Smith v. ABC Collections",
         "The debt collector called the consumer's workplace repeatedly after being told to stop. "
         "The collector used threatening language and disclosed the debt to coworkers. "
         "The court found violations of FDCPA sections 1692c and 1692d.",
         "district", "5th", "2022-01-15"),
        (2, "pkg2", "Jones v. Quick Recovery Inc",
         "Defendant sent collection letters that failed to include the required validation notice. "
         "The letters contained false representations about the amount owed. "
         "Summary judgment granted for plaintiff on FDCPA 1692e and 1692g claims.",
         "district", "5th", "2022-06-20"),
        (3, "pkg3", "Doe v. National Collectors",
         "Plaintiff alleged the debt collector called using an automatic telephone dialing system. "
         "The collector failed to obtain prior express consent before making robocalls. "
         "The court dismissed the TCPA claim for lack of standing.",
         "district", "9th", "2023-03-10"),
        (4, "pkg4", "Lee v. Debt Solutions LLC",
         "The collection agency continued calling after receiving a cease and desist letter. "
         "Defendant raised the bona fide error defense but failed to show procedures. "
         "Judgment for plaintiff with statutory damages of $1,000.",
         "district", "3rd", "2023-07-05"),
        (5, "pkg5", "Kim v. Credit Bureau Inc",
         "Consumer disputed the accuracy of information reported to credit bureaus. "
         "The furnisher failed to conduct a reasonable investigation after dispute. "
         "FCRA claim dismissed on summary judgment for defendant.",
         "district", "9th", "2023-11-01"),
        (6, "pkg6", "Park v. Recovery Associates",
         "Debt collector left voicemail messages that disclosed the debt to third parties. "
         "The messages did not contain the mini-Miranda warning required by FDCPA. "
         "Court awarded $1,000 statutory damages and $5,000 attorney fees.",
         "district", "5th", "2024-02-15"),
    ]

    chunks = []
    for oid, pkg, title, text_content, ct, circ, date in opinions_data:
        session.add(Opinion(
            id=oid, package_id=pkg, title=title, plain_text=text_content,
            court_type=ct, circuit=circ, date_issued=date,
        ))
        chunks.append({
            "chunk_id": f"{oid}_chunk_0", "opinion_id": oid, "chunk_index": 0,
            "text": text_content, "title": title, "court_name": f"Test Court",
            "court_type": ct, "circuit": circ, "date_issued": date,
            "statutes": "FDCPA" if oid != 5 else "FCRA",
            "predicted_outcome": "", "claim_sections": "",
        })
    session.commit()

    # Link opinions to statutes
    for oid in [1, 2, 3, 4, 6]:
        session.execute(sql_text(
            f"INSERT INTO opinion_statutes (opinion_id, statute_id) VALUES ({oid}, 1)"
        ))
    session.execute(sql_text(
        "INSERT INTO opinion_statutes (opinion_id, statute_id) VALUES (3, 2)"
    ))
    session.execute(sql_text(
        "INSERT INTO opinion_statutes (opinion_id, statute_id) VALUES (5, 1)"
    ))
    session.commit()

    # Add outcome predictions
    outcomes = [
        (1, "plaintiff_win", 0.85),
        (2, "plaintiff_win", 0.92),
        (3, "defendant_win", 0.78),
        (4, "plaintiff_win", 0.88),
        (5, "defendant_win", 0.82),
        (6, "plaintiff_win", 0.90),
    ]
    for oid, outcome, conf in outcomes:
        session.execute(sql_text(
            "INSERT INTO predictions (opinion_id, model_name, label_type, "
            "predicted_value, confidence) VALUES "
            f"({oid}, 'outcome_logreg_v1', 'outcome', '{outcome}', {conf})"
        ))
    session.commit()

    # Add entities
    entities = [
        (1, "DAMAGES_AWARDED", "$1,000"),
        (1, "ATTORNEY_FEES", "$3,000"),
        (1, "DEFENSE_TYPE", "bona fide error"),
        (2, "DAMAGES_AWARDED", "$2,500"),
        (2, "ATTORNEY_FEES", "$4,000"),
        (3, "DEFENSE_TYPE", "standing"),
        (3, "DEFENSE_TYPE", "mootness"),
        (4, "DAMAGES_AWARDED", "$1,000"),
        (4, "ATTORNEY_FEES", "$2,500"),
        (4, "DEFENSE_TYPE", "bona fide error"),
        (6, "DAMAGES_AWARDED", "$1,000"),
        (6, "ATTORNEY_FEES", "$5,000"),
    ]
    for oid, etype, evalue in entities:
        session.add(Entity(
            opinion_id=oid, entity_type=etype, entity_value=evalue,
            context_snippet="test context",
        ))
    session.commit()

    # Add claim type predictions
    claim_preds = [
        (1, "1692c"), (1, "1692d"),
        (2, "1692e"), (2, "1692g"),
        (4, "1692d"), (4, "1692e"),
        (6, "1692c"), (6, "1692d"),
    ]
    for oid, section in claim_preds:
        session.execute(sql_text(
            "INSERT INTO predictions (opinion_id, model_name, label_type, "
            "predicted_value, confidence) VALUES "
            f"({oid}, 'claim_type_logreg_v1', 'claim_type', '{section}', 0.7)"
        ))
    session.commit()
    session.close()

    # Build FAISS index
    texts = [c["text"] for c in chunks]
    embeddings = embed_chunks(texts)
    index = build_index()
    chunk_map = []
    add_to_index(index, chunk_map, chunks, embeddings)

    return engine, index, chunk_map


class TestWeightedOutcome:
    def test_majority_plaintiff_win(self, prediction_env):
        from predictor import evaluate_case
        engine, index, chunk_map = prediction_env
        # Query about FDCPA debt collection harassment — most similar cases are plaintiff wins
        result = evaluate_case(
            engine, index, chunk_map,
            "A debt collector called my workplace and told my boss about my debt",
        )
        assert result["predicted_outcome"] == "plaintiff_win"
        assert 0.0 <= result["confidence"] <= 1.0

    def test_returns_all_expected_keys(self, prediction_env):
        from predictor import evaluate_case
        engine, index, chunk_map = prediction_env
        result = evaluate_case(
            engine, index, chunk_map,
            "The collector violated the FDCPA by calling repeatedly",
        )
        expected_keys = {
            "predicted_outcome", "confidence",
            "damages_median", "damages_25th", "damages_75th",
            "fees_median", "fees_25th", "fees_75th",
            "similar_cases", "risk_factors", "claim_recommendations",
            "case_count",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_similar_cases_have_scores(self, prediction_env):
        from predictor import evaluate_case
        engine, index, chunk_map = prediction_env
        result = evaluate_case(
            engine, index, chunk_map,
            "debt collector called after cease and desist",
        )
        assert len(result["similar_cases"]) > 0
        for case in result["similar_cases"]:
            assert "similarity_score" in case
            assert "opinion_id" in case
            assert "title" in case
            assert "predicted_outcome" in case


class TestDamagesEstimation:
    def test_damages_returns_numbers(self, prediction_env):
        from predictor import evaluate_case
        engine, index, chunk_map = prediction_env
        result = evaluate_case(
            engine, index, chunk_map,
            "debt collector harassed me at work, FDCPA violation",
        )
        if result["damages_median"] is not None:
            assert result["damages_median"] > 0
            assert result["damages_25th"] is not None
            assert result["damages_75th"] is not None
            assert result["damages_25th"] <= result["damages_median"] <= result["damages_75th"]

    def test_fees_returns_numbers(self, prediction_env):
        from predictor import evaluate_case
        engine, index, chunk_map = prediction_env
        result = evaluate_case(
            engine, index, chunk_map,
            "FDCPA violation collection letters",
        )
        if result["fees_median"] is not None:
            assert result["fees_median"] > 0


class TestRiskFactors:
    def test_risk_factors_is_list(self, prediction_env):
        from predictor import evaluate_case
        engine, index, chunk_map = prediction_env
        result = evaluate_case(
            engine, index, chunk_map,
            "debt collection FDCPA violation",
        )
        assert isinstance(result["risk_factors"], list)

    def test_risk_factors_have_defense_and_count(self, prediction_env):
        from predictor import evaluate_case
        engine, index, chunk_map = prediction_env
        result = evaluate_case(
            engine, index, chunk_map,
            "TCPA robocall automatic dialer",
        )
        for rf in result["risk_factors"]:
            assert "defense_type" in rf
            assert "count" in rf


class TestClaimRecommendations:
    def test_claim_recommendations_is_list(self, prediction_env):
        from predictor import evaluate_case
        engine, index, chunk_map = prediction_env
        result = evaluate_case(
            engine, index, chunk_map,
            "debt collector called my workplace",
        )
        assert isinstance(result["claim_recommendations"], list)

    def test_claim_recommendations_have_fields(self, prediction_env):
        from predictor import evaluate_case
        engine, index, chunk_map = prediction_env
        result = evaluate_case(
            engine, index, chunk_map,
            "FDCPA debt collection harassing calls",
        )
        for rec in result["claim_recommendations"]:
            assert "claim_section" in rec
            assert "plaintiff_win_rate" in rec
            assert "count" in rec


class TestFiltering:
    def test_statute_filter(self, prediction_env):
        from predictor import evaluate_case
        engine, index, chunk_map = prediction_env
        result = evaluate_case(
            engine, index, chunk_map,
            "violation of consumer protection law",
            statute="FDCPA",
        )
        assert result["case_count"] > 0

    def test_circuit_filter(self, prediction_env):
        from predictor import evaluate_case
        engine, index, chunk_map = prediction_env
        result = evaluate_case(
            engine, index, chunk_map,
            "debt collection violation",
            circuit="5th",
        )
        for case in result["similar_cases"]:
            assert case["circuit"] == "5th"


class TestEdgeCases:
    def test_no_similar_cases(self, prediction_env):
        from predictor import evaluate_case
        engine, index, chunk_map = prediction_env
        # Very restrictive filter that matches nothing
        result = evaluate_case(
            engine, index, chunk_map,
            "completely unrelated topic about gardening",
            circuit="99th",
        )
        assert result["predicted_outcome"] is None
        assert result["confidence"] == 0.0
        assert result["similar_cases"] == []
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_predictor.py -v`
Expected: FAIL — `predictor` module not found

**Step 3: Implement predictor.py**

Create `predictor.py`:

```python
"""
Case Evaluation Predictor: predict outcome, damages, and recommendations
from similar precedents.

Usage:
    from predictor import evaluate_case
    result = evaluate_case(engine, index, chunk_map, "case summary text")
"""
import re
import logging
from collections import Counter

import numpy as np
from sqlalchemy import text

from search import search_opinions

logger = logging.getLogger(__name__)


def _parse_dollar(value):
    """Parse dollar string like '$1,000.00' to float."""
    if not value:
        return None
    cleaned = re.sub(r'[,$\s]', '', value)
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def _weighted_percentiles(values, weights, percentiles):
    """Compute weighted percentiles.

    Args:
        values: list of numeric values
        weights: list of corresponding weights (same length)
        percentiles: list of percentiles to compute (0-100)

    Returns list of percentile values.
    """
    if not values or not weights:
        return [None] * len(percentiles)

    values = np.array(values, dtype=float)
    weights = np.array(weights, dtype=float)

    # Sort by values
    sorted_idx = np.argsort(values)
    values = values[sorted_idx]
    weights = weights[sorted_idx]

    # Cumulative weight
    cum_weights = np.cumsum(weights)
    total_weight = cum_weights[-1]
    if total_weight == 0:
        return [None] * len(percentiles)

    cum_pct = cum_weights / total_weight * 100

    results = []
    for p in percentiles:
        idx = np.searchsorted(cum_pct, p)
        idx = min(idx, len(values) - 1)
        results.append(float(values[idx]))
    return results


def evaluate_case(engine, index, chunk_map, query_text,
                  statute=None, circuit=None, court_type=None, top_k=50):
    """Evaluate a case by finding similar precedents and computing predictions.

    Args:
        engine: SQLAlchemy engine
        index: FAISS index
        chunk_map: chunk metadata list
        query_text: case summary text
        statute: optional statute filter (FDCPA, TCPA, FCRA)
        circuit: optional circuit filter
        court_type: optional court type filter
        top_k: number of similar cases to retrieve

    Returns dict with:
        predicted_outcome, confidence, damages_median, damages_25th,
        damages_75th, fees_median, fees_25th, fees_75th,
        similar_cases, risk_factors, claim_recommendations, case_count
    """
    empty_result = {
        "predicted_outcome": None,
        "confidence": 0.0,
        "damages_median": None,
        "damages_25th": None,
        "damages_75th": None,
        "fees_median": None,
        "fees_25th": None,
        "fees_75th": None,
        "similar_cases": [],
        "risk_factors": [],
        "claim_recommendations": [],
        "case_count": 0,
    }

    if not query_text or not query_text.strip():
        return empty_result

    # Build filters
    filters = {}
    if statute:
        filters["statute"] = statute.upper()
    if circuit:
        filters["circuit"] = circuit
    if court_type:
        filters["court_type"] = court_type

    # Search for similar opinions
    similar = search_opinions(index, chunk_map, query_text,
                              top_k=top_k, filters=filters or None)

    if not similar:
        return empty_result

    opinion_ids = [s["opinion_id"] for s in similar]
    scores = {s["opinion_id"]: s["similarity_score"] for s in similar}

    # Load outcome predictions for similar opinions
    id_list = ",".join(str(oid) for oid in opinion_ids)
    with engine.connect() as conn:
        outcome_rows = conn.execute(text(
            f"SELECT opinion_id, predicted_value, confidence "
            f"FROM predictions "
            f"WHERE opinion_id IN ({id_list}) "
            f"AND label_type = 'outcome' AND model_name = 'outcome_logreg_v1'"
        )).fetchall()

        entity_rows = conn.execute(text(
            f"SELECT opinion_id, entity_type, entity_value "
            f"FROM entities "
            f"WHERE opinion_id IN ({id_list}) "
            f"AND entity_type IN ('DAMAGES_AWARDED', 'ATTORNEY_FEES', 'DEFENSE_TYPE')"
        )).fetchall()

        claim_rows = conn.execute(text(
            f"SELECT opinion_id, predicted_value "
            f"FROM predictions "
            f"WHERE opinion_id IN ({id_list}) "
            f"AND label_type = 'claim_type' AND model_name = 'claim_type_logreg_v1'"
        )).fetchall()

    # Build lookup maps
    outcomes = {r[0]: r[1] for r in outcome_rows}
    confidences = {r[0]: r[2] for r in outcome_rows}

    damages = {}
    fees = {}
    defenses = {}
    for oid, etype, evalue in entity_rows:
        if etype == "DAMAGES_AWARDED":
            amount = _parse_dollar(evalue)
            if amount is not None and amount > 0:
                damages[oid] = amount
        elif etype == "ATTORNEY_FEES":
            amount = _parse_dollar(evalue)
            if amount is not None and amount > 0:
                fees[oid] = amount
        elif etype == "DEFENSE_TYPE":
            defenses.setdefault(oid, []).append(evalue)

    claims = {}
    for oid, section in claim_rows:
        claims.setdefault(oid, []).append(section)

    # --- Weighted outcome prediction ---
    outcome_weights = {}  # outcome_value -> total weight
    for oid in opinion_ids:
        if oid in outcomes:
            outcome = outcomes[oid]
            weight = scores[oid]
            outcome_weights[outcome] = outcome_weights.get(outcome, 0) + weight

    if outcome_weights:
        total_weight = sum(outcome_weights.values())
        predicted_outcome = max(outcome_weights, key=outcome_weights.get)
        confidence = outcome_weights[predicted_outcome] / total_weight
    else:
        predicted_outcome = None
        confidence = 0.0

    # --- Weighted damages/fees estimation ---
    dmg_values = []
    dmg_weights = []
    for oid in opinion_ids:
        if oid in damages:
            dmg_values.append(damages[oid])
            dmg_weights.append(scores[oid])

    damages_25, damages_50, damages_75 = _weighted_percentiles(
        dmg_values, dmg_weights, [25, 50, 75]
    )

    fee_values = []
    fee_weights = []
    for oid in opinion_ids:
        if oid in fees:
            fee_values.append(fees[oid])
            fee_weights.append(scores[oid])

    fees_25, fees_50, fees_75 = _weighted_percentiles(
        fee_values, fee_weights, [25, 50, 75]
    )

    # --- Risk factors: defenses in defendant-win cases ---
    defense_in_losses = Counter()
    defense_total = Counter()
    for oid in opinion_ids:
        if oid in defenses:
            for d in defenses[oid]:
                defense_total[d] += 1
                if outcomes.get(oid) == "defendant_win":
                    defense_in_losses[d] += 1

    risk_factors = []
    for defense, loss_count in defense_in_losses.most_common():
        risk_factors.append({
            "defense_type": defense,
            "count": defense_total[defense],
            "loss_count": loss_count,
            "loss_rate": loss_count / defense_total[defense] if defense_total[defense] > 0 else 0,
        })

    # --- Claim recommendations: sections ranked by plaintiff win rate ---
    claim_stats = {}
    for oid in opinion_ids:
        if oid in claims:
            outcome = outcomes.get(oid)
            for section in claims[oid]:
                if section not in claim_stats:
                    claim_stats[section] = {"wins": 0, "total": 0, "damages_sum": 0, "damages_count": 0}
                claim_stats[section]["total"] += 1
                if outcome == "plaintiff_win":
                    claim_stats[section]["wins"] += 1
                if oid in damages:
                    claim_stats[section]["damages_sum"] += damages[oid]
                    claim_stats[section]["damages_count"] += 1

    claim_recommendations = []
    for section, stats in claim_stats.items():
        if stats["total"] >= 2:  # minimum occurrences
            avg_damages = (stats["damages_sum"] / stats["damages_count"]
                          if stats["damages_count"] > 0 else None)
            claim_recommendations.append({
                "claim_section": section,
                "count": stats["total"],
                "plaintiff_win_rate": stats["wins"] / stats["total"],
                "avg_damages": avg_damages,
            })

    claim_recommendations.sort(key=lambda x: x["plaintiff_win_rate"], reverse=True)

    # --- Build similar cases list ---
    similar_cases = []
    for s in similar:
        oid = s["opinion_id"]
        similar_cases.append({
            "opinion_id": oid,
            "title": s["title"],
            "circuit": s["circuit"],
            "court_type": s["court_type"],
            "date_issued": s["date_issued"],
            "similarity_score": s["similarity_score"],
            "predicted_outcome": outcomes.get(oid, "unknown"),
            "damages": damages.get(oid),
            "attorney_fees": fees.get(oid),
        })

    return {
        "predicted_outcome": predicted_outcome,
        "confidence": float(confidence),
        "damages_median": damages_50,
        "damages_25th": damages_25,
        "damages_75th": damages_75,
        "fees_median": fees_50,
        "fees_25th": fees_25,
        "fees_75th": fees_75,
        "similar_cases": similar_cases,
        "risk_factors": risk_factors,
        "claim_recommendations": claim_recommendations,
        "case_count": len(similar),
    }
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_predictor.py -v`
Expected: All tests pass

**Step 5: Commit**

```bash
git add predictor.py tests/test_predictor.py
git commit -m "feat: add case evaluation predictor module"
```

---

### Task 2: Create Predictor Dashboard

**Files:**
- Create: `pages/7_Predictor.py`

**Step 1: Create the dashboard page**

Create `pages/7_Predictor.py`:

```python
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
    "**For research and prioritization only. Not legal advice.** "
    "Always review underlying precedents before drawing conclusions."
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

if index is None:
    st.error("No FAISS index found. Run `python pipeline.py` first.")
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

    top_k = st.slider("Similar Cases to Analyze", 10, 100, 50)

# --- Input Section ---
st.subheader("Describe Your Case")
query_text = st.text_area(
    "Case Summary",
    height=200,
    placeholder=(
        "Describe the facts of your case. For example:\n\n"
        "A debt collector called my workplace three times after I sent a written "
        "cease and desist letter. The collector told my supervisor about the debt. "
        "The original creditor was Chase Bank for a credit card balance of $5,000. "
        "I am in the 5th Circuit."
    ),
)

evaluate_btn = st.button("Evaluate Case", type="primary", disabled=not query_text)

if evaluate_btn and query_text:
    with st.spinner("Analyzing similar cases..."):
        result = evaluate_case(
            engine, index, chunk_map, query_text,
            statute=selected_statute if selected_statute != "All" else None,
            circuit=selected_circuit if selected_circuit != "All" else None,
            court_type=selected_court_type if selected_court_type != "All" else None,
            top_k=top_k,
        )

    if result["predicted_outcome"] is None:
        st.warning("No similar cases found with the selected filters. Try broadening your search.")
        st.stop()

    # --- Summary Metrics ---
    st.divider()
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        outcome_display = result["predicted_outcome"].replace("_", " ").title()
        st.metric("Predicted Outcome", outcome_display)
        st.caption(f"Confidence: {result['confidence']:.0%}")

    with col2:
        if result["damages_median"] is not None:
            st.metric("Est. Damages (Median)", f"${result['damages_median']:,.0f}")
            st.caption(
                f"Range: ${result['damages_25th']:,.0f} – ${result['damages_75th']:,.0f}"
            )
        else:
            st.metric("Est. Damages", "N/A")

    with col3:
        if result["fees_median"] is not None:
            st.metric("Est. Attorney Fees (Median)", f"${result['fees_median']:,.0f}")
            st.caption(
                f"Range: ${result['fees_25th']:,.0f} – ${result['fees_75th']:,.0f}"
            )
        else:
            st.metric("Est. Attorney Fees", "N/A")

    with col4:
        st.metric("Similar Cases Found", result["case_count"])

    # --- Tabs ---
    tab1, tab2, tab3 = st.tabs([
        "Similar Precedents", "Risk Factors", "Claim Recommendations",
    ])

    # === Tab 1: Similar Precedents ===
    with tab1:
        if result["similar_cases"]:
            cases_df = pd.DataFrame(result["similar_cases"])
            display_df = cases_df.head(20).copy()
            display_df["similarity_score"] = display_df["similarity_score"].apply(
                lambda x: f"{x:.3f}"
            )
            display_df["damages"] = display_df["damages"].apply(
                lambda x: f"${x:,.0f}" if pd.notna(x) and x is not None else "N/A"
            )
            display_df["attorney_fees"] = display_df["attorney_fees"].apply(
                lambda x: f"${x:,.0f}" if pd.notna(x) and x is not None else "N/A"
            )
            display_df["predicted_outcome"] = display_df["predicted_outcome"].apply(
                lambda x: x.replace("_", " ").title() if x else "Unknown"
            )
            st.dataframe(
                display_df[["title", "circuit", "date_issued", "predicted_outcome",
                            "damages", "attorney_fees", "similarity_score"]],
                use_container_width=True, hide_index=True,
                column_config={
                    "title": st.column_config.TextColumn("Case", width="large"),
                    "similarity_score": st.column_config.TextColumn("Score", width="small"),
                },
            )

    # === Tab 2: Risk Factors ===
    with tab2:
        if result["risk_factors"]:
            st.subheader("Defense Types in Losing Cases")
            rf_df = pd.DataFrame(result["risk_factors"])
            rf_df["loss_rate"] = rf_df["loss_rate"].apply(lambda x: f"{x:.0%}")

            st.dataframe(
                rf_df[["defense_type", "count", "loss_count", "loss_rate"]],
                use_container_width=True, hide_index=True,
                column_config={
                    "defense_type": "Defense Type",
                    "count": "Times Raised",
                    "loss_count": "Led to Loss",
                    "loss_rate": "Loss Rate",
                },
            )

            # Bar chart
            fig = px.bar(
                rf_df, x="defense_type", y="count",
                labels={"defense_type": "Defense", "count": "Times Raised"},
                title="Defense Frequency in Similar Cases",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No defense-related risk factors identified in similar cases.")

    # === Tab 3: Claim Recommendations ===
    with tab3:
        if result["claim_recommendations"]:
            st.subheader("Claim Sections by Success Rate")
            cr_df = pd.DataFrame(result["claim_recommendations"])
            cr_df["plaintiff_win_rate"] = cr_df["plaintiff_win_rate"].apply(
                lambda x: f"{x:.0%}"
            )
            cr_df["avg_damages"] = cr_df["avg_damages"].apply(
                lambda x: f"${x:,.0f}" if pd.notna(x) and x is not None else "N/A"
            )
            cr_df["claim_section"] = cr_df["claim_section"].apply(
                lambda x: f"§{x}"
            )
            st.dataframe(
                cr_df[["claim_section", "count", "plaintiff_win_rate", "avg_damages"]],
                use_container_width=True, hide_index=True,
                column_config={
                    "claim_section": "Section",
                    "count": "Cases",
                    "plaintiff_win_rate": "Win Rate",
                    "avg_damages": "Avg Damages",
                },
            )
        else:
            st.info("Not enough claim data in similar cases for recommendations.")
```

**Step 2: Verify dashboard compiles**

Run: `python -c "import py_compile; py_compile.compile('pages/7_Predictor.py', doraise=True)"`
Expected: No errors

**Step 3: Commit**

```bash
git add pages/7_Predictor.py
git commit -m "feat: add Case Evaluator dashboard page"
```

---

### Task 3: Run full test suite and final validation

**Files:** None (validation only)

**Step 1: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All tests pass (150 existing + ~15 new predictor tests)

**Step 2: Verify all dashboard pages compile**

Run: `python -c "import py_compile; [py_compile.compile(f'pages/{f}', doraise=True) for f in ['1_Search.py','2_Topics.py','3_Analytics.py','4_Citations.py','5_Entities.py','6_Trends.py','7_Predictor.py']]; print('All OK')"`
Expected: "All OK"

**Step 3: Commit if any fixes needed**

Only commit if fixes were required.
