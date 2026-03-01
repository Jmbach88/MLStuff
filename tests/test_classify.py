import os
import pytest
import numpy as np

os.environ["ML_LOCAL_DB"] = ":memory:"

from db import get_local_engine, init_local_db, get_session, Opinion, Label
from sqlalchemy import text


def _setup_labeled_data():
    """Create opinions with labels for training."""
    engine = get_local_engine()
    init_local_db(engine)
    session = get_session(engine)

    session.execute(text("DELETE FROM labels"))
    session.execute(text("DELETE FROM opinions"))
    session.commit()

    opinions = []
    labels = []
    oid = 1
    for outcome in ["plaintiff_win", "defendant_win", "mixed"]:
        for i in range(30):
            if outcome == "plaintiff_win":
                txt = f"The court grants judgment for plaintiff {oid}. Defendant is liable under §1692e for debt collection abuse."
            elif outcome == "defendant_win":
                txt = f"Defendant's motion is granted {oid}. Plaintiff's complaint is dismissed under §227 of the TCPA."
            else:
                txt = f"Motion granted in part and denied in part {oid}. Claims under §1681s-2 partially succeed."
            opinions.append(Opinion(id=oid, package_id=f"p{oid}", title=f"Case {oid}", plain_text=txt))
            labels.append(Label(opinion_id=oid, label_type="outcome", label_value=outcome, source="regex", confidence=0.9))
            oid += 1

    session.add_all(opinions)
    session.add_all(labels)
    session.commit()
    return engine


class TestOutcomeClassifier:
    def test_train_outcome_model(self):
        from classify import train_outcome_model
        engine = _setup_labeled_data()
        result = train_outcome_model(engine)
        assert "accuracy" in result
        assert "f1_score" in result
        assert result["accuracy"] > 0.3
        assert result["model_name"] is not None

    def test_predict_outcomes(self):
        from classify import train_outcome_model, predict_outcomes
        engine = _setup_labeled_data()
        train_outcome_model(engine)
        count = predict_outcomes(engine)
        assert count > 0


class TestClaimTypeClassifier:
    def test_train_claim_type_model(self):
        from classify import train_claim_type_model
        engine = _setup_labeled_data()

        session = get_session(engine)
        from label import label_claim_types
        opinions = session.execute(text("SELECT id, plain_text FROM opinions")).fetchall()
        for oid, txt in opinions:
            sections = label_claim_types(txt)
            for s in sections:
                session.add(Label(opinion_id=oid, label_type="claim_type", label_value=s, source="regex", confidence=0.95))
        session.commit()
        session.close()

        result = train_claim_type_model(engine)
        assert "sections_trained" in result

    def test_predict_claim_types(self):
        from classify import train_claim_type_model, predict_claim_types
        engine = _setup_labeled_data()

        session = get_session(engine)
        from label import label_claim_types
        opinions = session.execute(text("SELECT id, plain_text FROM opinions")).fetchall()
        for oid, txt in opinions:
            sections = label_claim_types(txt)
            for s in sections:
                session.add(Label(opinion_id=oid, label_type="claim_type", label_value=s, source="regex", confidence=0.95))
        session.commit()
        session.close()

        train_claim_type_model(engine)
        count = predict_claim_types(engine)
        assert count > 0
