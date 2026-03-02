"""
Topic modeling pipeline using BERTopic on opinion embeddings.

Usage:
    python topics.py              # fit topics and store assignments
    python topics.py --refit      # clear old assignments and refit
    python topics.py --info       # print topic summary
"""
import argparse
import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
from sqlalchemy import text

import config
from db import get_local_engine, init_local_db, get_session, Prediction, Model as ModelRecord

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(config.PROJECT_ROOT, "data", "models")

MODEL_NAME = "bertopic_v1"
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, "bertopic_v1.pkl")
DEFAULT_COORDS_PATH = os.path.join(MODELS_DIR, "topic_coords_v1.npz")


def aggregate_opinion_embeddings(chunk_map, vectors):
    """Groups chunk vectors by opinion_id, averages them, L2-normalizes.

    Args:
        chunk_map: list of dicts with opinion_id, title, court_name, etc.
        vectors: numpy array of shape (n_chunks, dim)

    Returns:
        (opinion_ids, embeddings, metadata) where:
            opinion_ids: list of ints
            embeddings: numpy array (n_opinions, dim)
            metadata: dict mapping opinion_id to {title, court_name, court_type, circuit, date_issued, statutes}
    """
    groups = defaultdict(list)
    meta = {}

    for i, entry in enumerate(chunk_map):
        oid = entry["opinion_id"]
        groups[oid].append(vectors[i])
        if oid not in meta:
            meta[oid] = {
                "title": entry.get("title", ""),
                "court_name": entry.get("court_name", ""),
                "court_type": entry.get("court_type", ""),
                "circuit": entry.get("circuit", ""),
                "date_issued": entry.get("date_issued", ""),
                "statutes": entry.get("statutes", ""),
            }

    opinion_ids = sorted(groups.keys())
    embeddings = []
    for oid in opinion_ids:
        vecs = np.array(groups[oid], dtype=np.float32)
        avg = vecs.mean(axis=0)
        norm = np.linalg.norm(avg)
        if norm > 0:
            avg = avg / norm
        embeddings.append(avg)

    embeddings = np.array(embeddings, dtype=np.float32)
    return opinion_ids, embeddings, meta


def fit_topics(opinion_ids, embeddings, docs, model_path=None,
               umap_model=None, hdbscan_model=None):
    """Fits BERTopic with UMAP and HDBSCAN on opinion embeddings.

    Args:
        opinion_ids: list of opinion IDs
        embeddings: numpy array (n_opinions, dim)
        docs: list of document strings for c-TF-IDF
        model_path: path to save the BERTopic model (default: data/models/bertopic_v1.pkl)
        umap_model: optional pre-configured UMAP model
        hdbscan_model: optional pre-configured HDBSCAN model

    Returns:
        (topics, probs, topic_info)
    """
    from umap import UMAP
    from fast_hdbscan import HDBSCAN
    from bertopic import BERTopic

    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    if umap_model is None:
        umap_model = UMAP(
            n_components=5,
            n_neighbors=15,
            min_dist=0.0,
            metric="cosine",
            random_state=42,
        )

    if hdbscan_model is None:
        hdbscan_model = HDBSCAN(
            min_cluster_size=15,
            min_samples=5,
            metric="euclidean",
            prediction_data=False,
        )

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        verbose=True,
    )

    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    topic_model.save(model_path, serialization="pickle", save_ctfidf=True)

    topic_info = topic_model.get_topic_info()
    logger.info(f"Fitted BERTopic: {len(set(topics))} topics, saved to {model_path}")
    return topics, probs, topic_info


def store_topic_assignments(engine, opinion_ids, topics, probs):
    """Stores topic assignments as Prediction rows.

    Clears existing predictions with model_name='bertopic_v1' and label_type='topic',
    then stores new rows.

    Args:
        engine: SQLAlchemy engine
        opinion_ids: list of opinion IDs
        topics: list of topic assignments (ints)
        probs: list/array of confidence values
    """
    session = get_session(engine)
    try:
        session.query(Prediction).filter(
            Prediction.model_name == MODEL_NAME,
            Prediction.label_type == "topic",
        ).delete()
        session.commit()

        now = datetime.now(timezone.utc).isoformat()
        predictions = []
        for oid, topic, prob in zip(opinion_ids, topics, probs):
            conf = float(prob) if prob is not None else None
            predictions.append(Prediction(
                opinion_id=oid,
                model_name=MODEL_NAME,
                label_type="topic",
                predicted_value=f"topic_{topic}",
                confidence=conf,
                created_at=now,
            ))

        session.add_all(predictions)
        session.commit()
        logger.info(f"Stored {len(predictions)} topic predictions")
    finally:
        session.close()


def save_2d_coords(opinion_ids, embeddings, path=None):
    """Runs UMAP to 2D and saves coordinates.

    Args:
        opinion_ids: list of opinion IDs
        embeddings: numpy array (n_opinions, dim)
        path: output path (default: data/models/topic_coords_v1.npz)
    """
    from umap import UMAP

    if path is None:
        path = DEFAULT_COORDS_PATH

    reducer = UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    coords = reducer.fit_transform(embeddings)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, opinion_ids=np.array(opinion_ids), coords=coords)
    logger.info(f"Saved 2D coordinates to {path}")


def get_topic_summary(engine):
    """Queries predictions table grouped by predicted_value for topic type.

    Returns list of rows with topic and count.
    """
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT predicted_value, COUNT(*) as count, AVG(confidence) as avg_confidence "
            "FROM predictions "
            "WHERE model_name = :model_name AND label_type = 'topic' "
            "GROUP BY predicted_value "
            "ORDER BY count DESC"
        ), {"model_name": MODEL_NAME})
        rows = result.fetchall()
    return rows


def run_topic_modeling(engine=None, refit=False):
    """Full topic modeling pipeline.

    1. Loads FAISS index and extracts vectors
    2. Aggregates to opinion level
    3. Gets first chunk text per opinion for c-TF-IDF
    4. Fits BERTopic
    5. Stores assignments
    6. Saves 2D coordinates
    7. Saves ModelRecord to DB
    """
    from index import load_index

    if engine is None:
        engine = get_local_engine()
        init_local_db(engine)

    # Load FAISS index
    faiss_index, chunk_map = load_index()
    if faiss_index is None:
        logger.error("No FAISS index found. Run embed.py first.")
        return

    n_vectors = faiss_index.ntotal
    dim = faiss_index.d
    logger.info(f"Loaded {n_vectors} vectors of dim {dim}")

    # Extract all vectors
    vectors = np.array([faiss_index.reconstruct(i) for i in range(n_vectors)], dtype=np.float32)

    # Aggregate to opinion level
    opinion_ids, embeddings, metadata = aggregate_opinion_embeddings(chunk_map, vectors)
    logger.info(f"Aggregated to {len(opinion_ids)} opinions")

    # Get first chunk text per opinion for c-TF-IDF
    first_texts = {}
    for entry in chunk_map:
        oid = entry["opinion_id"]
        if oid not in first_texts:
            first_texts[oid] = entry.get("text", "")
    docs = [first_texts.get(oid, "") for oid in opinion_ids]

    # Fit topics
    if refit or not os.path.exists(DEFAULT_MODEL_PATH):
        topics, probs, topic_info = fit_topics(opinion_ids, embeddings, docs)
    else:
        logger.info("Model already exists. Use --refit to retrain.")
        return

    # Store assignments
    store_topic_assignments(engine, opinion_ids, topics, probs)

    # Save 2D coordinates
    save_2d_coords(opinion_ids, embeddings)

    # Save ModelRecord
    session = get_session(engine)
    try:
        existing = session.query(ModelRecord).filter_by(name=MODEL_NAME).first()
        now = datetime.now(timezone.utc).isoformat()
        if existing:
            existing.trained_at = now
            existing.params_json = json.dumps({
                "n_opinions": len(opinion_ids),
                "n_topics": len(set(topics)),
                "umap_dim": 5,
                "min_cluster_size": 15,
            })
        else:
            session.add(ModelRecord(
                name=MODEL_NAME,
                label_type="topic",
                trained_at=now,
                params_json=json.dumps({
                    "n_opinions": len(opinion_ids),
                    "n_topics": len(set(topics)),
                    "umap_dim": 5,
                    "min_cluster_size": 15,
                }),
            ))
        session.commit()
    finally:
        session.close()

    logger.info("Topic modeling pipeline complete.")


def main():
    parser = argparse.ArgumentParser(description="Topic modeling pipeline")
    parser.add_argument("--refit", action="store_true", help="Clear and refit topics")
    parser.add_argument("--info", action="store_true", help="Print topic summary")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    engine = get_local_engine()
    init_local_db(engine)

    if args.info:
        rows = get_topic_summary(engine)
        if not rows:
            print("No topic assignments found.")
        else:
            print(f"{'Topic':<20} {'Count':>8} {'Avg Confidence':>15}")
            print("-" * 45)
            for row in rows:
                print(f"{row[0]:<20} {row[1]:>8} {row[2]:>15.4f}")
        return

    run_topic_modeling(engine=engine, refit=args.refit)


if __name__ == "__main__":
    main()
