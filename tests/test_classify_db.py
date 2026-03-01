import os
import pytest

os.environ["ML_LOCAL_DB"] = ":memory:"

from db import get_local_engine, init_local_db, get_session, Label, Prediction, Model


def test_label_table_created():
    engine = get_local_engine()
    init_local_db(engine)
    session = get_session(engine)

    label = Label(
        opinion_id=1,
        label_type="outcome",
        label_value="plaintiff_win",
        source="regex",
        confidence=0.9,
    )
    session.add(label)
    session.commit()

    result = session.query(Label).filter_by(opinion_id=1).first()
    assert result.label_value == "plaintiff_win"
    assert result.source == "regex"
    session.close()


def test_prediction_table_created():
    engine = get_local_engine()
    init_local_db(engine)
    session = get_session(engine)

    pred = Prediction(
        opinion_id=1,
        model_name="outcome_logreg_v1",
        label_type="outcome",
        predicted_value="defendant_win",
        confidence=0.85,
        created_at="2026-03-01T00:00:00",
    )
    session.add(pred)
    session.commit()

    result = session.query(Prediction).filter_by(opinion_id=1).first()
    assert result.predicted_value == "defendant_win"
    session.close()


def test_model_table_created():
    engine = get_local_engine()
    init_local_db(engine)
    session = get_session(engine)

    model = Model(
        name="outcome_logreg_v1",
        label_type="outcome",
        accuracy=0.82,
        f1_score=0.78,
        trained_at="2026-03-01T00:00:00",
        params_json='{"max_iter": 1000}',
    )
    session.add(model)
    session.commit()

    result = session.query(Model).filter_by(name="outcome_logreg_v1").first()
    assert result.accuracy == 0.82
    session.close()
