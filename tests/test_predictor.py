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
