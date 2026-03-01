"""
Train, evaluate, and predict with text classifiers.

Usage:
    python classify.py                  # train all models and predict
    python classify.py --train-only     # just train, don't predict
    python classify.py --predict-only   # predict with existing models
"""
import argparse
import json
import logging
import os
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sqlalchemy import text

import config
from db import get_local_engine, init_local_db, get_session, Label, Prediction, Model as ModelRecord

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(config.PROJECT_ROOT, "data", "models")

MIN_EXAMPLES = 10


def _ensure_models_dir():
    """Create models directory if it doesn't exist."""
    os.makedirs(MODELS_DIR, exist_ok=True)


def train_outcome_model(engine=None):
    """Train a logistic regression model for outcome classification.

    Loads labeled opinions with outcome labels, does 80/20 stratified split,
    trains TfidfVectorizer + LogisticRegression, evaluates, saves model.

    Returns dict with model_name, accuracy, f1_score, report, train_size, test_size.
    """
    if engine is None:
        engine = get_local_engine()
        init_local_db(engine)

    session = get_session(engine)

    rows = session.execute(text(
        "SELECT o.id, o.plain_text, l.label_value "
        "FROM opinions o JOIN labels l ON o.id = l.opinion_id "
        "WHERE l.label_type = 'outcome' AND o.plain_text IS NOT NULL AND o.plain_text != ''"
    )).fetchall()
    session.close()

    if len(rows) < 10:
        logger.warning(f"Only {len(rows)} labeled opinions, need at least 10")
        return {"error": "insufficient data", "count": len(rows)}

    texts = [r[1] for r in rows]
    labels = [r[2] for r in rows]

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42
    )

    tfidf = TfidfVectorizer(max_features=50000, sublinear_tf=True, stop_words="english")
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, output_dict=True)

    model_name = "outcome_logreg_v1"
    _ensure_models_dir()
    model_path = os.path.join(MODELS_DIR, "outcome_model.pkl")
    joblib.dump({"tfidf": tfidf, "clf": clf}, model_path)
    logger.info(f"Saved outcome model to {model_path}")

    # Save model record to DB
    session = get_session(engine)
    existing = session.query(ModelRecord).filter_by(name=model_name).first()
    if existing:
        existing.accuracy = float(acc)
        existing.f1_score = float(f1)
        existing.trained_at = datetime.now(timezone.utc).isoformat()
        existing.params_json = json.dumps({"max_features": 50000, "max_iter": 1000})
    else:
        session.add(ModelRecord(
            name=model_name,
            label_type="outcome",
            accuracy=float(acc),
            f1_score=float(f1),
            trained_at=datetime.now(timezone.utc).isoformat(),
            params_json=json.dumps({"max_features": 50000, "max_iter": 1000}),
        ))
    session.commit()
    session.close()

    result = {
        "model_name": model_name,
        "accuracy": float(acc),
        "f1_score": float(f1),
        "report": report,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }
    logger.info(f"Outcome model trained: accuracy={acc:.3f}, f1={f1:.3f}")
    return result


def predict_outcomes(engine=None):
    """Predict outcomes for opinions without existing predictions.

    Loads the trained model from disk, gets opinions without predictions,
    predicts in batches, stores Prediction records with confidence.

    Returns count of predictions made.
    """
    if engine is None:
        engine = get_local_engine()
        init_local_db(engine)

    model_path = os.path.join(MODELS_DIR, "outcome_model.pkl")
    if not os.path.exists(model_path):
        logger.error("No trained outcome model found")
        return 0

    model_data = joblib.load(model_path)
    tfidf = model_data["tfidf"]
    clf = model_data["clf"]
    model_name = "outcome_logreg_v1"

    session = get_session(engine)

    rows = session.execute(text(
        "SELECT o.id, o.plain_text FROM opinions o "
        "WHERE o.plain_text IS NOT NULL AND o.plain_text != '' "
        "AND o.id NOT IN (SELECT opinion_id FROM predictions WHERE label_type = 'outcome')"
    )).fetchall()

    if not rows:
        logger.info("No opinions to predict")
        session.close()
        return 0

    count = 0
    batch_size = 5000
    now = datetime.now(timezone.utc).isoformat()

    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        ids = [r[0] for r in batch]
        texts = [r[1] for r in batch]

        X = tfidf.transform(texts)
        preds = clf.predict(X)
        probas = clf.predict_proba(X)

        for j, (oid, pred_label) in enumerate(zip(ids, preds)):
            class_idx = list(clf.classes_).index(pred_label)
            conf = float(probas[j][class_idx])
            session.add(Prediction(
                opinion_id=oid,
                model_name=model_name,
                label_type="outcome",
                predicted_value=pred_label,
                confidence=conf,
                created_at=now,
            ))
            count += 1

        session.commit()

    session.close()
    logger.info(f"Predicted outcomes for {count} opinions")
    return count


def train_claim_type_model(engine=None):
    """Train a multi-label classifier for claim type sections.

    Gets sections with >= MIN_EXAMPLES examples, builds multi-label dataset
    with MultiLabelBinarizer, trains TfidfVectorizer + OneVsRestClassifier.

    Returns dict with model_name, sections_trained, per_section_metrics, avg_f1.
    """
    if engine is None:
        engine = get_local_engine()
        init_local_db(engine)

    session = get_session(engine)

    # Get section counts to filter by MIN_EXAMPLES
    section_counts = session.execute(text(
        "SELECT label_value, COUNT(*) as cnt FROM labels "
        "WHERE label_type = 'claim_type' GROUP BY label_value HAVING cnt >= :min_ex"
    ), {"min_ex": MIN_EXAMPLES}).fetchall()

    if not section_counts:
        logger.warning("No claim type sections with enough examples")
        session.close()
        return {"error": "insufficient data", "sections_trained": 0}

    valid_sections = {r[0] for r in section_counts}
    logger.info(f"Training on {len(valid_sections)} sections: {valid_sections}")

    # Build multi-label dataset: for each opinion, gather all claim_type labels
    rows = session.execute(text(
        "SELECT o.id, o.plain_text FROM opinions o "
        "WHERE o.plain_text IS NOT NULL AND o.plain_text != '' "
        "AND o.id IN (SELECT DISTINCT opinion_id FROM labels WHERE label_type = 'claim_type')"
    )).fetchall()

    opinion_labels = {}
    for oid, txt in rows:
        opinion_labels[oid] = {"text": txt, "sections": []}

    label_rows = session.execute(text(
        "SELECT opinion_id, label_value FROM labels WHERE label_type = 'claim_type'"
    )).fetchall()
    session.close()

    for oid, val in label_rows:
        if oid in opinion_labels and val in valid_sections:
            opinion_labels[oid]["sections"].append(val)

    # Filter to opinions that have at least one valid section
    data = [(v["text"], v["sections"]) for v in opinion_labels.values() if v["sections"]]

    if len(data) < 10:
        logger.warning(f"Only {len(data)} opinions with valid claim types")
        return {"error": "insufficient data", "sections_trained": 0}

    texts = [d[0] for d in data]
    label_lists = [d[1] for d in data]

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(label_lists)

    X_train, X_test, y_train, y_test = train_test_split(
        texts, Y, test_size=0.2, random_state=42
    )

    tfidf = TfidfVectorizer(max_features=50000, sublinear_tf=True, stop_words="english")
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=42))
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)

    # Per-section metrics
    per_section_metrics = {}
    section_f1s = []
    for i, section in enumerate(mlb.classes_):
        if y_test[:, i].sum() > 0:
            sec_f1 = f1_score(y_test[:, i], y_pred[:, i], zero_division=0)
            per_section_metrics[section] = {"f1": float(sec_f1), "support": int(y_test[:, i].sum())}
            section_f1s.append(sec_f1)

    avg_f1 = float(np.mean(section_f1s)) if section_f1s else 0.0

    model_name = "claim_type_ovr_v1"
    _ensure_models_dir()
    model_path = os.path.join(MODELS_DIR, "claim_type_model.pkl")
    joblib.dump({"tfidf": tfidf, "clf": clf, "mlb": mlb, "sections": list(mlb.classes_)}, model_path)
    logger.info(f"Saved claim type model to {model_path}")

    # Save model record
    session = get_session(engine)
    existing = session.query(ModelRecord).filter_by(name=model_name).first()
    if existing:
        existing.f1_score = avg_f1
        existing.trained_at = datetime.now(timezone.utc).isoformat()
        existing.params_json = json.dumps({"sections": list(mlb.classes_), "max_features": 50000})
    else:
        session.add(ModelRecord(
            name=model_name,
            label_type="claim_type",
            f1_score=avg_f1,
            trained_at=datetime.now(timezone.utc).isoformat(),
            params_json=json.dumps({"sections": list(mlb.classes_), "max_features": 50000}),
        ))
    session.commit()
    session.close()

    result = {
        "model_name": model_name,
        "sections_trained": len(mlb.classes_),
        "per_section_metrics": per_section_metrics,
        "avg_f1": avg_f1,
    }
    logger.info(f"Claim type model trained: {len(mlb.classes_)} sections, avg_f1={avg_f1:.3f}")
    return result


def predict_claim_types(engine=None):
    """Predict claim types for opinions without existing predictions.

    Loads multi-label model, predicts using inverse_transform,
    uses decision_function for confidence scores.

    Returns count of predictions made.
    """
    if engine is None:
        engine = get_local_engine()
        init_local_db(engine)

    model_path = os.path.join(MODELS_DIR, "claim_type_model.pkl")
    if not os.path.exists(model_path):
        logger.error("No trained claim type model found")
        return 0

    model_data = joblib.load(model_path)
    tfidf = model_data["tfidf"]
    clf = model_data["clf"]
    mlb = model_data["mlb"]
    model_name = "claim_type_ovr_v1"

    session = get_session(engine)

    rows = session.execute(text(
        "SELECT o.id, o.plain_text FROM opinions o "
        "WHERE o.plain_text IS NOT NULL AND o.plain_text != '' "
        "AND o.id NOT IN (SELECT opinion_id FROM predictions WHERE label_type = 'claim_type')"
    )).fetchall()

    if not rows:
        logger.info("No opinions to predict claim types for")
        session.close()
        return 0

    count = 0
    batch_size = 5000
    now = datetime.now(timezone.utc).isoformat()

    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        ids = [r[0] for r in batch]
        texts = [r[1] for r in batch]

        X = tfidf.transform(texts)
        preds_binary = clf.predict(X)
        preds_labels = mlb.inverse_transform(preds_binary)

        # Get decision function scores for confidence
        try:
            decision_scores = clf.decision_function(X)
            if decision_scores.ndim == 1:
                decision_scores = decision_scores.reshape(-1, 1)
        except AttributeError:
            decision_scores = None

        for j, (oid, pred_sections) in enumerate(zip(ids, preds_labels)):
            for section in pred_sections:
                conf = 0.5
                if decision_scores is not None:
                    sec_idx = list(mlb.classes_).index(section)
                    # Sigmoid of decision function as confidence
                    raw = float(decision_scores[j][sec_idx])
                    conf = float(1.0 / (1.0 + np.exp(-raw)))

                session.add(Prediction(
                    opinion_id=oid,
                    model_name=model_name,
                    label_type="claim_type",
                    predicted_value=section,
                    confidence=conf,
                    created_at=now,
                ))
                count += 1

        session.commit()

    session.close()
    logger.info(f"Predicted claim types for {count} opinion-section pairs")
    return count


def update_chunk_map_with_predictions(engine=None):
    """Add predicted_outcome and claim_sections to existing FAISS chunk_map."""
    if engine is None:
        engine = get_local_engine()

    from index import load_index, save_index

    index, chunk_map = load_index()
    if index is None:
        logger.error("No FAISS index found")
        return 0

    session = get_session(engine)

    outcome_rows = session.execute(text(
        "SELECT opinion_id, predicted_value FROM predictions "
        "WHERE model_name = 'outcome_logreg_v1' AND label_type = 'outcome'"
    )).fetchall()
    outcome_map = {r[0]: r[1] for r in outcome_rows}

    claim_rows = session.execute(text(
        "SELECT opinion_id, predicted_value FROM predictions "
        "WHERE model_name = 'claim_type_logreg_v1' AND label_type = 'claim_type'"
    )).fetchall()
    claim_map = {}
    for oid, section in claim_rows:
        claim_map.setdefault(oid, []).append(section)

    session.close()

    updated = 0
    for entry in chunk_map:
        oid = entry["opinion_id"]
        entry["predicted_outcome"] = outcome_map.get(oid, "")
        sections = claim_map.get(oid, [])
        entry["claim_sections"] = ",".join(sorted(sections))
        updated += 1

    save_index(index, chunk_map)
    logger.info(f"Updated {updated} chunk_map entries with predictions")
    return updated


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Train and predict with classifiers")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--update-index", action="store_true", help="Update FAISS chunk_map with predictions")
    args = parser.parse_args()

    engine = get_local_engine()
    init_local_db(engine)

    if not args.predict_only:
        print("\n=== Training Outcome Model ===")
        outcome_result = train_outcome_model(engine)
        print(f"Accuracy: {outcome_result.get('accuracy', 'N/A')}")
        print(f"Macro F1: {outcome_result.get('f1_score', 'N/A')}")
        print("\n=== Training Claim Type Model ===")
        claim_result = train_claim_type_model(engine)
        print(f"Sections trained: {claim_result.get('sections_trained', 0)}")
        print(f"Avg F1: {claim_result.get('avg_f1', 'N/A')}")

    if not args.train_only:
        print("\n=== Predicting Outcomes ===")
        n = predict_outcomes(engine)
        print(f"Predicted {n} opinions")
        print("\n=== Predicting Claim Types ===")
        n = predict_claim_types(engine)
        print(f"Predicted {n} opinions")

    if args.update_index:
        print("\n=== Updating FAISS Index Metadata ===")
        n = update_chunk_map_with_predictions(engine)
        print(f"Updated {n} chunk_map entries")
