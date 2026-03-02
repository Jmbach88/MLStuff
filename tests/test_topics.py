import os
import pytest
import numpy as np

os.environ["ML_LOCAL_DB"] = ":memory:"

from db import get_local_engine, init_local_db, get_session, Prediction, Opinion


def _make_chunk_map_and_vectors():
    """Create test data: 2 opinions with 5 chunks total (3 + 2)."""
    dim = 384
    rng = np.random.RandomState(42)

    chunk_map = [
        {"opinion_id": 1, "chunk_index": 0, "title": "Case A", "court_name": "District Court",
         "court_type": "district", "circuit": "9th", "date_issued": "2023-01-01", "statutes": "1692e",
         "text": "Plaintiff alleges violations."},
        {"opinion_id": 1, "chunk_index": 1, "title": "Case A", "court_name": "District Court",
         "court_type": "district", "circuit": "9th", "date_issued": "2023-01-01", "statutes": "1692e",
         "text": "The court finds for plaintiff."},
        {"opinion_id": 1, "chunk_index": 2, "title": "Case A", "court_name": "District Court",
         "court_type": "district", "circuit": "9th", "date_issued": "2023-01-01", "statutes": "1692e",
         "text": "Damages are awarded."},
        {"opinion_id": 2, "chunk_index": 0, "title": "Case B", "court_name": "Appeals Court",
         "court_type": "appeals", "circuit": "2nd", "date_issued": "2023-06-15", "statutes": "1692f",
         "text": "Defendant appeals the ruling."},
        {"opinion_id": 2, "chunk_index": 1, "title": "Case B", "court_name": "Appeals Court",
         "court_type": "appeals", "circuit": "2nd", "date_issued": "2023-06-15", "statutes": "1692f",
         "text": "Appeal is denied."},
    ]

    vectors = rng.randn(5, dim).astype(np.float32)
    return chunk_map, vectors


class TestEmbeddingAggregation:
    def test_aggregate_produces_one_vector_per_opinion(self):
        from topics import aggregate_opinion_embeddings
        chunk_map, vectors = _make_chunk_map_and_vectors()
        opinion_ids, embeddings, metadata = aggregate_opinion_embeddings(chunk_map, vectors)
        assert len(opinion_ids) == 2
        assert embeddings.shape == (2, 384)

    def test_aggregate_normalizes_vectors(self):
        from topics import aggregate_opinion_embeddings
        chunk_map, vectors = _make_chunk_map_and_vectors()
        opinion_ids, embeddings, metadata = aggregate_opinion_embeddings(chunk_map, vectors)
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_aggregate_preserves_metadata(self):
        from topics import aggregate_opinion_embeddings
        chunk_map, vectors = _make_chunk_map_and_vectors()
        opinion_ids, embeddings, metadata = aggregate_opinion_embeddings(chunk_map, vectors)
        assert 1 in metadata
        assert 2 in metadata
        assert metadata[1]["title"] == "Case A"
        assert metadata[1]["court_name"] == "District Court"
        assert metadata[1]["circuit"] == "9th"
        assert metadata[2]["title"] == "Case B"
        assert metadata[2]["court_type"] == "appeals"
        assert metadata[2]["statutes"] == "1692f"


class TestTopicFitting:
    def test_fit_topics_returns_results(self):
        from umap import UMAP
        from fast_hdbscan import HDBSCAN
        from topics import fit_topics

        rng = np.random.RandomState(42)
        dim = 384
        # Create 3 well-separated clusters of 20 vectors each
        cluster1 = rng.randn(20, dim).astype(np.float32) + 10.0
        cluster2 = rng.randn(20, dim).astype(np.float32) - 10.0
        cluster3 = rng.randn(20, dim).astype(np.float32)
        embeddings = np.vstack([cluster1, cluster2, cluster3])
        # L2-normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        opinion_ids = list(range(60))
        docs = [f"Document about topic {i % 3} with content {i}" for i in range(60)]

        umap_model = UMAP(n_components=5, n_neighbors=15, min_dist=0.0,
                          metric="cosine", random_state=42)
        hdbscan_model = HDBSCAN(min_cluster_size=5, min_samples=3,
                                metric="euclidean", prediction_data=False)

        topics, probs, topic_info = fit_topics(
            opinion_ids, embeddings, docs,
            model_path=os.path.join(os.path.dirname(__file__), "tmp_model.pkl"),
            umap_model=umap_model, hdbscan_model=hdbscan_model,
        )

        assert len(topics) == 60
        # Clean up
        tmp = os.path.join(os.path.dirname(__file__), "tmp_model.pkl")
        if os.path.exists(tmp):
            os.remove(tmp)

    def test_fit_topics_saves_model(self, tmp_path):
        from umap import UMAP
        from fast_hdbscan import HDBSCAN
        from topics import fit_topics

        rng = np.random.RandomState(42)
        dim = 384
        cluster1 = rng.randn(20, dim).astype(np.float32) + 10.0
        cluster2 = rng.randn(20, dim).astype(np.float32) - 10.0
        cluster3 = rng.randn(20, dim).astype(np.float32)
        embeddings = np.vstack([cluster1, cluster2, cluster3])
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        opinion_ids = list(range(60))
        docs = [f"Document about topic {i % 3} with content {i}" for i in range(60)]
        model_path = str(tmp_path / "test_bertopic.pkl")

        umap_model = UMAP(n_components=5, n_neighbors=15, min_dist=0.0,
                          metric="cosine", random_state=42)
        hdbscan_model = HDBSCAN(min_cluster_size=5, min_samples=3,
                                metric="euclidean", prediction_data=False)

        fit_topics(opinion_ids, embeddings, docs, model_path=model_path,
                   umap_model=umap_model, hdbscan_model=hdbscan_model)

        assert os.path.exists(model_path)


class TestTopicStorage:
    def test_store_topic_assignments(self):
        from topics import store_topic_assignments

        engine = get_local_engine()
        init_local_db(engine)
        session = get_session(engine)

        # Create test opinions
        from sqlalchemy import text as sql_text
        session.execute(sql_text("DELETE FROM predictions"))
        session.execute(sql_text("DELETE FROM opinions"))
        session.commit()

        for oid in [1, 2, 3]:
            session.add(Opinion(id=oid, package_id=f"pkg_{oid}", title=f"Case {oid}"))
        session.commit()
        session.close()

        opinion_ids = [1, 2, 3]
        topics = [0, 1, 0]
        probs = [0.95, 0.80, 0.70]

        store_topic_assignments(engine, opinion_ids, topics, probs)

        session = get_session(engine)
        preds = session.query(Prediction).filter(
            Prediction.model_name == "bertopic_v1",
            Prediction.label_type == "topic",
        ).all()

        assert len(preds) == 3
        values = {p.opinion_id: p.predicted_value for p in preds}
        assert values[1] == "topic_0"
        assert values[2] == "topic_1"
        assert values[3] == "topic_0"

        confidences = {p.opinion_id: p.confidence for p in preds}
        assert abs(confidences[1] - 0.95) < 1e-6
        session.close()
