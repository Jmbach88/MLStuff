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

# Legal domain stop words — common terms that appear in all opinions
# and don't help distinguish topics from each other
LEGAL_STOP_WORDS = [
    "court", "plaintiff", "defendant", "motion", "order", "case",
    "complaint", "filed", "claim", "claims", "action", "party", "parties",
    "relief", "judgment", "granted", "denied", "pursuant", "alleged",
    "alleges", "argues", "argue", "argument", "asserting", "asserts",
    "contends", "states", "stated", "facts", "fact", "evidence",
    "standard", "review", "record", "matter", "issue", "issues",
    "whether", "however", "therefore", "moreover", "thus", "also",
    "would", "must", "shall", "may", "could", "upon", "thereof",
    "herein", "therein", "supra", "infra",
    "judge", "district", "circuit", "appeals", "appellate",
    "federal", "united", "states", "section", "subsection",
    "statute", "statutory", "act", "rule", "rules", "regulation",
    "amended", "amend", "amendment",
    "counsel", "attorney", "attorneys", "law", "legal",
    "cause", "causes", "hearing", "trial", "proceeding", "proceedings",
]


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
               umap_model=None, hdbscan_model=None, nr_topics=None,
               vectorizer_model=None, reduce_outliers=True):
    """Fits BERTopic with UMAP and HDBSCAN on opinion embeddings.

    Args:
        opinion_ids: list of opinion IDs
        embeddings: numpy array (n_opinions, dim)
        docs: list of document strings for c-TF-IDF
        model_path: path to save the BERTopic model (default: data/models/bertopic_v1.pkl)
        umap_model: optional pre-configured UMAP model
        hdbscan_model: optional pre-configured HDBSCAN model
        nr_topics: optional target number of topics (merges similar topics)
        vectorizer_model: optional CountVectorizer for c-TF-IDF
        reduce_outliers: whether to reassign outliers to nearest topic (default True)

    Returns:
        (topics, probs, topic_info)
    """
    from umap import UMAP
    from fast_hdbscan import HDBSCAN
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer

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

    if vectorizer_model is None:
        vectorizer_model = CountVectorizer(
            ngram_range=(1, 2),
            stop_words=LEGAL_STOP_WORDS,
            min_df=5,
            max_df=0.95,
        )

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        nr_topics=nr_topics,
        verbose=True,
    )

    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)

    # Reduce outliers by reassigning to nearest topic
    if reduce_outliers:
        n_outliers_before = sum(1 for t in topics if t == -1)
        if n_outliers_before > 0:
            topics = topic_model.reduce_outliers(docs, topics, strategy="embeddings", embeddings=embeddings)
            topic_model.update_topics(docs, topics=topics, vectorizer_model=vectorizer_model)
            n_outliers_after = sum(1 for t in topics if t == -1)
            logger.info(f"Outlier reduction: {n_outliers_before} -> {n_outliers_after}")

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


def run_topic_modeling(engine=None, refit=False, nr_topics=None):
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

    # Get text per opinion for c-TF-IDF (concatenate first 3 chunks)
    opinion_texts = defaultdict(list)
    for entry in chunk_map:
        oid = entry["opinion_id"]
        if len(opinion_texts[oid]) < 3:
            opinion_texts[oid].append(entry.get("text", ""))
    docs = [" ".join(opinion_texts.get(oid, [""])) for oid in opinion_ids]

    # Fit topics
    if refit or not os.path.exists(DEFAULT_MODEL_PATH):
        topics, probs, topic_info = fit_topics(
            opinion_ids, embeddings, docs, nr_topics=nr_topics,
        )
    else:
        logger.info("Model already exists. Use --refit to retrain.")
        return

    n_topics = len(set(t for t in topics if t != -1))
    n_outliers = sum(1 for t in topics if t == -1)
    logger.info(f"Found {n_topics} topics, {n_outliers} outliers")

    # Store assignments
    store_topic_assignments(engine, opinion_ids, topics, probs)

    # Save 2D coordinates
    save_2d_coords(opinion_ids, embeddings)

    # Save ModelRecord with topic words
    top_topics = topic_info.head(min(n_topics, 50)).to_dict("records") if topic_info is not None else []
    session = get_session(engine)
    try:
        existing = session.query(ModelRecord).filter_by(name=MODEL_NAME).first()
        now = datetime.now(timezone.utc).isoformat()
        params = {
            "n_opinions": len(opinion_ids),
            "n_topics": n_topics,
            "n_outliers": n_outliers,
            "nr_topics": nr_topics,
            "top_topics": top_topics,
        }
        if existing:
            existing.trained_at = now
            existing.params_json = json.dumps(params, default=str)
        else:
            session.add(ModelRecord(
                name=MODEL_NAME,
                label_type="topic",
                trained_at=now,
                params_json=json.dumps(params, default=str),
            ))
        session.commit()
    finally:
        session.close()

    logger.info("Topic modeling pipeline complete.")


def relabel_topics(engine=None, model_name=None):
    """Use Ollama LLM to generate clean topic labels from keywords.

    Loads the saved BERTopic model, extracts keywords per topic,
    sends them to Ollama via OpenAI-compatible API, and stores
    the clean labels in the models table params_json.

    Args:
        engine: SQLAlchemy engine (default: creates one)
        model_name: Ollama model to use (default: config.OLLAMA_MODEL)
    """
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("openai package required for relabeling. Install with: pip install openai")
        return

    if engine is None:
        engine = get_local_engine()
        init_local_db(engine)

    if model_name is None:
        model_name = config.OLLAMA_MODEL

    # Load saved BERTopic model
    if not os.path.exists(DEFAULT_MODEL_PATH):
        logger.error("No BERTopic model found. Run `python topics.py --refit` first.")
        return

    from bertopic import BERTopic
    topic_model = BERTopic.load(DEFAULT_MODEL_PATH)

    topic_info = topic_model.get_topic_info()
    # Filter out outlier topic (-1)
    topic_ids = [t for t in topic_info["Topic"].tolist() if t != -1]

    if not topic_ids:
        logger.warning("No topics found in model.")
        return

    logger.info(f"Relabeling {len(topic_ids)} topics using {model_name}...")

    client = OpenAI(
        base_url=config.OLLAMA_OPENAI_BASE,
        api_key="ollama",  # Ollama doesn't require a real key
    )

    custom_labels = {}
    for i, tid in enumerate(topic_ids):
        keywords = topic_model.get_topic(tid)
        if not keywords:
            continue

        # keywords is list of (word, score) tuples — take top 10
        top_words = [w for w, _ in keywords[:10]]
        keyword_str = ", ".join(top_words)

        prompt = (
            f"These keywords describe a topic found in federal court opinions "
            f"about debt collection and consumer protection law:\n\n"
            f"Keywords: {keyword_str}\n\n"
            f"Generate a short, descriptive label (2-5 words) for this topic. "
            f"Return ONLY the label, nothing else."
        )

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=30,
            )
            label = response.choices[0].message.content.strip().strip('"').strip("'")
            custom_labels[tid] = label
            logger.info(f"  [{i+1}/{len(topic_ids)}] Topic {tid}: {label}")
        except Exception as e:
            logger.warning(f"  [{i+1}/{len(topic_ids)}] Topic {tid}: LLM error: {e}")
            continue

    if not custom_labels:
        logger.warning("No labels generated.")
        return

    # Update params_json in DB with custom_labels
    session = get_session(engine)
    try:
        model_record = session.query(ModelRecord).filter_by(name=MODEL_NAME).first()
        if model_record and model_record.params_json:
            params = json.loads(model_record.params_json)
        else:
            params = {}

        params["custom_labels"] = {str(k): v for k, v in custom_labels.items()}
        params["relabel_model"] = model_name
        params["relabeled_at"] = datetime.now(timezone.utc).isoformat()

        if model_record:
            model_record.params_json = json.dumps(params, default=str)
        session.commit()
        logger.info(f"Stored {len(custom_labels)} custom topic labels.")
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description="Topic modeling pipeline")
    parser.add_argument("--refit", action="store_true", help="Clear and refit topics")
    parser.add_argument("--info", action="store_true", help="Print topic summary")
    parser.add_argument("--nr-topics", type=int, default=None, help="Merge to target number of topics")
    parser.add_argument("--relabel", action="store_true",
                        help="Use LLM to generate clean topic labels")
    parser.add_argument("--relabel-model", type=str, default=None,
                        help="Ollama model for relabeling (default: config.OLLAMA_MODEL)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    engine = get_local_engine()
    init_local_db(engine)

    if args.info:
        rows = get_topic_summary(engine)
        if not rows:
            print("No topic assignments found.")
        else:
            # Load custom labels if available
            session = get_session(engine)
            model_record = session.query(ModelRecord).filter_by(name=MODEL_NAME).first()
            session.close()
            custom_labels = {}
            if model_record and model_record.params_json:
                params = json.loads(model_record.params_json)
                custom_labels = params.get("custom_labels", {})

            print(f"{'Topic':<20} {'Count':>8} {'Avg Conf':>10} {'Label'}")
            print("-" * 70)
            for row in rows:
                topic_num = row[0].replace("topic_", "")
                label = custom_labels.get(topic_num, "")
                print(f"{row[0]:<20} {row[1]:>8} {row[2]:>10.4f} {label}")
        return

    if args.relabel:
        relabel_topics(engine=engine, model_name=args.relabel_model)
        return

    run_topic_modeling(engine=engine, refit=args.refit, nr_topics=args.nr_topics)


if __name__ == "__main__":
    main()
