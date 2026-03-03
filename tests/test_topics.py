import importlib
import json
import os
from unittest.mock import patch, MagicMock
import pytest
import numpy as np

os.environ["ML_LOCAL_DB"] = ":memory:"

from db import get_local_engine, init_local_db, get_session, Prediction, Opinion
from db import Model as ModelRecord


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

        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer_model = CountVectorizer(ngram_range=(1, 1), min_df=1)

        topics, probs, topic_info = fit_topics(
            opinion_ids, embeddings, docs,
            model_path=os.path.join(os.path.dirname(__file__), "tmp_model.pkl"),
            umap_model=umap_model, hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model, reduce_outliers=False,
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

        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer_model = CountVectorizer(ngram_range=(1, 1), min_df=1)

        fit_topics(opinion_ids, embeddings, docs, model_path=model_path,
                   umap_model=umap_model, hdbscan_model=hdbscan_model,
                   vectorizer_model=vectorizer_model, reduce_outliers=False)

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


class TestRelabelTopics:
    """Tests for LLM-based topic relabeling."""

    def _setup_model_record(self, engine):
        """Create a model record with topic info for testing."""
        session = get_session(engine)
        from sqlalchemy import text as sql_text
        session.execute(sql_text("DELETE FROM models"))
        session.commit()

        params = {
            "n_opinions": 100,
            "n_topics": 3,
            "n_outliers": 5,
            "top_topics": [
                {"Topic": 0, "Name": "0_debt_collection_agency"},
                {"Topic": 1, "Name": "1_credit_card_dispute"},
                {"Topic": 2, "Name": "2_statute_limitations_defense"},
            ],
        }
        session.add(ModelRecord(
            name="bertopic_v1",
            label_type="topic",
            trained_at="2026-01-01T00:00:00",
            params_json=json.dumps(params),
        ))
        session.commit()
        session.close()

    def _mock_bertopic_model(self):
        """Create a mock BERTopic model with topic keywords."""
        mock_model = MagicMock()
        mock_model.get_topic_info.return_value = MagicMock()
        mock_model.get_topic_info.return_value.__getitem__ = MagicMock(
            side_effect=lambda key: {
                "Topic": [-1, 0, 1, 2],
            }[key]
        )
        # get_topic_info()["Topic"].tolist()
        topic_series = MagicMock()
        topic_series.tolist.return_value = [-1, 0, 1, 2]
        mock_model.get_topic_info.return_value.__getitem__ = MagicMock(return_value=topic_series)

        mock_model.get_topic.side_effect = lambda tid: {
            0: [("debt", 0.9), ("collection", 0.8), ("agency", 0.7)],
            1: [("credit", 0.9), ("card", 0.8), ("dispute", 0.7)],
            2: [("statute", 0.9), ("limitations", 0.8), ("defense", 0.7)],
        }.get(tid, [])
        return mock_model

    def _mock_openai_response(self, label):
        """Create a mock OpenAI chat completion response."""
        mock_choice = MagicMock()
        mock_choice.message.content = label
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        return mock_response

    def _patch_relabel_deps(self, mock_model, mock_client):
        """Context manager helper to patch BERTopic and OpenAI inside relabel_topics."""
        mock_bertopic_cls = MagicMock()
        mock_bertopic_cls.load.return_value = mock_model
        mock_openai_cls = MagicMock(return_value=mock_client)

        # Patch the imports inside relabel_topics
        import importlib
        mock_openai_module = MagicMock()
        mock_openai_module.OpenAI = mock_openai_cls
        mock_bertopic_module = MagicMock()
        mock_bertopic_module.BERTopic = mock_bertopic_cls

        return patch.dict("sys.modules", {
            "openai": mock_openai_module,
            "bertopic": mock_bertopic_module,
        })

    def test_relabel_stores_custom_labels(self):
        import topics
        importlib.reload(topics)  # reload to pick up fresh module state

        engine = get_local_engine()
        init_local_db(engine)
        self._setup_model_record(engine)

        mock_model = self._mock_bertopic_model()
        mock_client = MagicMock()
        labels = iter(["Debt Collection Practices", "Credit Card Disputes", "Statute of Limitations"])
        mock_client.chat.completions.create.side_effect = lambda **kwargs: self._mock_openai_response(next(labels))

        with self._patch_relabel_deps(mock_model, mock_client), \
             patch("topics.os.path.exists", return_value=True):
            importlib.reload(topics)
            topics.relabel_topics(engine=engine, model_name="llama3.1:latest")

        session = get_session(engine)
        record = session.query(ModelRecord).filter_by(name="bertopic_v1").first()
        params = json.loads(record.params_json)
        session.close()

        assert "custom_labels" in params
        assert params["custom_labels"]["0"] == "Debt Collection Practices"
        assert params["custom_labels"]["1"] == "Credit Card Disputes"
        assert params["custom_labels"]["2"] == "Statute of Limitations"
        assert params["relabel_model"] == "llama3.1:latest"
        assert "relabeled_at" in params

    def test_relabel_preserves_existing_params(self):
        import topics
        engine = get_local_engine()
        init_local_db(engine)
        self._setup_model_record(engine)

        mock_model = self._mock_bertopic_model()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_openai_response("Test Label")

        with self._patch_relabel_deps(mock_model, mock_client), \
             patch("topics.os.path.exists", return_value=True):
            importlib.reload(topics)
            topics.relabel_topics(engine=engine)

        session = get_session(engine)
        record = session.query(ModelRecord).filter_by(name="bertopic_v1").first()
        params = json.loads(record.params_json)
        session.close()

        assert params["n_topics"] == 3
        assert params["n_opinions"] == 100
        assert len(params["top_topics"]) == 3
        assert "custom_labels" in params

    def test_relabel_handles_llm_error(self):
        import topics
        engine = get_local_engine()
        init_local_db(engine)
        self._setup_model_record(engine)

        mock_model = self._mock_bertopic_model()
        mock_client = MagicMock()

        call_count = 0
        def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise Exception("Connection refused")
            return self._mock_openai_response("Good Label")

        mock_client.chat.completions.create.side_effect = side_effect

        with self._patch_relabel_deps(mock_model, mock_client), \
             patch("topics.os.path.exists", return_value=True):
            importlib.reload(topics)
            topics.relabel_topics(engine=engine)

        session = get_session(engine)
        record = session.query(ModelRecord).filter_by(name="bertopic_v1").first()
        params = json.loads(record.params_json)
        session.close()

        assert len(params["custom_labels"]) == 2
        assert "0" in params["custom_labels"]
        assert "2" in params["custom_labels"]

    def test_relabel_no_model_file(self):
        """relabel_topics returns early if no model file exists."""
        import topics

        engine = get_local_engine()
        init_local_db(engine)

        mock_openai_module = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_openai_module}), \
             patch("topics.os.path.exists", return_value=False):
            importlib.reload(topics)
            # Should not raise
            topics.relabel_topics(engine=engine)

    def test_relabel_strips_quotes_from_response(self):
        import topics
        engine = get_local_engine()
        init_local_db(engine)
        self._setup_model_record(engine)

        mock_model = self._mock_bertopic_model()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_openai_response('"Debt Collection"')

        with self._patch_relabel_deps(mock_model, mock_client), \
             patch("topics.os.path.exists", return_value=True):
            importlib.reload(topics)
            topics.relabel_topics(engine=engine)

        session = get_session(engine)
        record = session.query(ModelRecord).filter_by(name="bertopic_v1").first()
        params = json.loads(record.params_json)
        session.close()

        assert params["custom_labels"]["0"] == "Debt Collection"
