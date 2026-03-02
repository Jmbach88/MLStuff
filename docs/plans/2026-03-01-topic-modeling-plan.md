# Topic Modeling Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Discover natural topics across 30K federal court opinions using BERTopic with pre-computed embeddings, store assignments, and build an interactive Topics dashboard.

**Architecture:** Load existing FAISS chunk embeddings, average them per opinion to get 30K opinion-level vectors. Fit BERTopic (UMAP + HDBSCAN + c-TF-IDF) on these vectors. Store topic assignments in the `predictions` table and 2D coordinates in a .npz file. Build a Streamlit dashboard with topic table, bar chart, scatter plot, and temporal evolution.

**Tech Stack:** BERTopic, UMAP, HDBSCAN, Plotly, NumPy, SQLAlchemy, Streamlit

---

### Task 1: Add dependencies

**Files:**
- Modify: `requirements.txt`

**Step 1: Add BERTopic and visualization dependencies to requirements.txt**

Add these lines to the end of `requirements.txt`:

```
bertopic>=0.16.0
umap-learn>=0.5.0
hdbscan>=0.8.33
plotly>=5.0.0
```

**Step 2: Install dependencies**

Run: `python -m pip install -r requirements.txt`
Expected: All packages install successfully.

**Step 3: Verify imports work**

Run: `python -c "from bertopic import BERTopic; from umap import UMAP; import hdbscan; import plotly; print('OK')"`
Expected: Prints `OK`

**Step 4: Commit**

```bash
git add requirements.txt
git commit -m "feat: add BERTopic, UMAP, HDBSCAN, Plotly dependencies"
```

---

### Task 2: Embedding aggregation and topic fitting

**Files:**
- Create: `topics.py`
- Test: `tests/test_topics.py`

**Step 1: Write the failing tests**

Create `tests/test_topics.py`:

```python
"""Tests for topic modeling pipeline."""
import os
import numpy as np
import pytest


class TestEmbeddingAggregation:
    def test_aggregate_produces_one_vector_per_opinion(self):
        from topics import aggregate_opinion_embeddings

        # Simulate chunk_map: 2 opinions, 3 chunks for opinion 1, 2 chunks for opinion 2
        chunk_map = [
            {"opinion_id": 1, "title": "Case A", "court_name": "District",
             "court_type": "district", "circuit": "5th", "date_issued": "2020-01-01",
             "statutes": "FDCPA"},
            {"opinion_id": 1, "title": "Case A", "court_name": "District",
             "court_type": "district", "circuit": "5th", "date_issued": "2020-01-01",
             "statutes": "FDCPA"},
            {"opinion_id": 1, "title": "Case A", "court_name": "District",
             "court_type": "district", "circuit": "5th", "date_issued": "2020-01-01",
             "statutes": "FDCPA"},
            {"opinion_id": 2, "title": "Case B", "court_name": "Appeals",
             "court_type": "circuit", "circuit": "9th", "date_issued": "2021-06-15",
             "statutes": "TCPA"},
            {"opinion_id": 2, "title": "Case B", "court_name": "Appeals",
             "court_type": "circuit", "circuit": "9th", "date_issued": "2021-06-15",
             "statutes": "TCPA"},
        ]
        # 5 chunk vectors, dim=4 for simplicity
        vectors = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
        ], dtype=np.float32)

        opinion_ids, embeddings, metadata = aggregate_opinion_embeddings(chunk_map, vectors)

        assert len(opinion_ids) == 2
        assert embeddings.shape == (2, 4)
        assert 1 in opinion_ids
        assert 2 in opinion_ids

    def test_aggregate_normalizes_vectors(self):
        from topics import aggregate_opinion_embeddings

        chunk_map = [
            {"opinion_id": 1, "title": "", "court_name": "", "court_type": "",
             "circuit": "", "date_issued": "", "statutes": ""},
            {"opinion_id": 1, "title": "", "court_name": "", "court_type": "",
             "circuit": "", "date_issued": "", "statutes": ""},
        ]
        vectors = np.array([[3, 0, 0, 0], [0, 4, 0, 0]], dtype=np.float32)

        opinion_ids, embeddings, metadata = aggregate_opinion_embeddings(chunk_map, vectors)

        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_aggregate_preserves_metadata(self):
        from topics import aggregate_opinion_embeddings

        chunk_map = [
            {"opinion_id": 42, "title": "Smith v. Jones", "court_name": "SDNY",
             "court_type": "district", "circuit": "2nd", "date_issued": "2022-03-15",
             "statutes": "FDCPA,FCRA"},
        ]
        vectors = np.array([[1, 0, 0, 0]], dtype=np.float32)

        opinion_ids, embeddings, metadata = aggregate_opinion_embeddings(chunk_map, vectors)

        assert metadata[42]["title"] == "Smith v. Jones"
        assert metadata[42]["circuit"] == "2nd"
        assert metadata[42]["statutes"] == "FDCPA,FCRA"


class TestTopicFitting:
    def test_fit_topics_returns_results(self):
        from topics import fit_topics

        np.random.seed(42)
        # Create 3 synthetic clusters of 20 vectors each
        cluster1 = np.random.randn(20, 384).astype(np.float32) + np.array([5, 0] + [0]*382)
        cluster2 = np.random.randn(20, 384).astype(np.float32) + np.array([0, 5] + [0]*382)
        cluster3 = np.random.randn(20, 384).astype(np.float32) + np.array([-5, -5] + [0]*382)
        embeddings = np.vstack([cluster1, cluster2, cluster3])
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        opinion_ids = list(range(60))
        # Simple docs for c-TF-IDF
        docs = (["debt collector violation fdcpa"] * 20 +
                ["robocall telephone tcpa consumer"] * 20 +
                ["credit report fcra dispute"] * 20)

        topics, probs, topic_info = fit_topics(opinion_ids, embeddings, docs)

        assert len(topics) == 60
        assert topic_info is not None
        # Should find at least 2 topics (may vary with small data)
        unique_topics = set(t for t in topics if t != -1)
        assert len(unique_topics) >= 1

    def test_fit_topics_saves_model(self, tmp_path):
        from topics import fit_topics
        import config

        np.random.seed(42)
        embeddings = np.random.randn(60, 384).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        opinion_ids = list(range(60))
        docs = ["legal opinion text"] * 60

        model_path = str(tmp_path / "bertopic_test.pkl")
        topics, probs, topic_info = fit_topics(
            opinion_ids, embeddings, docs, model_path=model_path
        )

        assert os.path.exists(model_path)


class TestTopicStorage:
    def test_store_topic_assignments(self):
        from topics import store_topic_assignments
        from db import get_local_engine, init_local_db, get_session, Prediction

        os.environ["ML_LOCAL_DB"] = ":memory:"
        engine = get_local_engine()
        init_local_db(engine)

        opinion_ids = [1, 2, 3]
        topics = [0, 1, -1]
        probs = np.array([0.9, 0.8, 0.0])

        store_topic_assignments(engine, opinion_ids, topics, probs)

        session = get_session(engine)
        preds = session.query(Prediction).filter_by(label_type="topic").all()
        assert len(preds) == 3
        assert preds[0].predicted_value == "topic_0"
        assert preds[2].predicted_value == "topic_-1"
        session.close()
        del os.environ["ML_LOCAL_DB"]
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_topics.py -v`
Expected: All tests FAIL with `ModuleNotFoundError: No module named 'topics'`

**Step 3: Implement topics.py**

Create `topics.py`:

```python
"""
Topic modeling with BERTopic using pre-computed embeddings.

Usage:
    python topics.py              # fit topics and store assignments
    python topics.py --refit      # clear old assignments, refit
    python topics.py --info       # print topic summary
"""
import argparse
import json
import logging
import os
from datetime import datetime, timezone

import numpy as np
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP

import config
from db import get_local_engine, init_local_db, get_session, Prediction, Model as ModelRecord
from index import load_index

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(config.PROJECT_ROOT, "data", "models")


def aggregate_opinion_embeddings(chunk_map, vectors):
    """Average chunk embeddings per opinion, L2-normalize.

    Args:
        chunk_map: list of dicts with opinion_id and metadata per chunk
        vectors: numpy array of shape (n_chunks, dim)

    Returns:
        (opinion_ids, embeddings, metadata) where:
        - opinion_ids: list of ints
        - embeddings: numpy array (n_opinions, dim)
        - metadata: dict mapping opinion_id -> {title, court_name, ...}
    """
    opinion_chunks = {}
    opinion_meta = {}

    for i, entry in enumerate(chunk_map):
        oid = entry["opinion_id"]
        if oid not in opinion_chunks:
            opinion_chunks[oid] = []
            opinion_meta[oid] = {
                "title": entry.get("title", ""),
                "court_name": entry.get("court_name", ""),
                "court_type": entry.get("court_type", ""),
                "circuit": entry.get("circuit", ""),
                "date_issued": entry.get("date_issued", ""),
                "statutes": entry.get("statutes", ""),
            }
        opinion_chunks[oid].append(vectors[i])

    opinion_ids = sorted(opinion_chunks.keys())
    embeddings = np.array([
        np.mean(opinion_chunks[oid], axis=0) for oid in opinion_ids
    ], dtype=np.float32)

    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms

    return opinion_ids, embeddings, opinion_meta


def fit_topics(opinion_ids, embeddings, docs, model_path=None):
    """Fit BERTopic on pre-computed embeddings.

    Args:
        opinion_ids: list of opinion IDs
        embeddings: numpy array (n, dim)
        docs: list of document strings (for c-TF-IDF word extraction)
        model_path: optional path to save fitted model

    Returns:
        (topics, probs, topic_info) where:
        - topics: list of topic assignments per opinion
        - probs: numpy array of probabilities
        - topic_info: DataFrame with topic details
    """
    umap_model = UMAP(
        n_components=5, n_neighbors=15, min_dist=0.0,
        metric="cosine", random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=15, min_samples=5,
        metric="euclidean", prediction_data=True,
    )

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        verbose=True,
    )

    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)

    topic_info = topic_model.get_topic_info()

    if model_path is None:
        os.makedirs(MODELS_DIR, exist_ok=True)
        model_path = os.path.join(MODELS_DIR, "bertopic_v1.pkl")

    topic_model.save(model_path, serialization="pickle", save_ctfidf=True)
    logger.info(f"Saved BERTopic model to {model_path}")

    return topics, probs, topic_info


def store_topic_assignments(engine, opinion_ids, topics, probs):
    """Store topic assignments in the predictions table.

    Clears existing topic predictions first, then inserts new ones.
    """
    session = get_session(engine)

    # Clear old topic predictions
    session.query(Prediction).filter_by(
        label_type="topic", model_name="bertopic_v1"
    ).delete()

    now = datetime.now(timezone.utc).isoformat()
    for oid, topic, prob in zip(opinion_ids, topics, probs):
        session.add(Prediction(
            opinion_id=oid,
            model_name="bertopic_v1",
            label_type="topic",
            predicted_value=f"topic_{topic}",
            confidence=float(prob) if prob is not None else 0.0,
            created_at=now,
        ))

    session.commit()
    session.close()
    logger.info(f"Stored {len(opinion_ids)} topic assignments")


def save_2d_coords(opinion_ids, embeddings, path=None):
    """Run UMAP to 2D and save coordinates for scatter plot."""
    if path is None:
        path = os.path.join(MODELS_DIR, "topic_coords_v1.npz")

    umap_2d = UMAP(
        n_components=2, n_neighbors=15, min_dist=0.1,
        metric="cosine", random_state=42,
    )
    coords = umap_2d.fit_transform(embeddings)
    np.savez(path, opinion_ids=np.array(opinion_ids), coords=coords)
    logger.info(f"Saved 2D coordinates to {path}")
    return coords


def get_topic_summary(engine):
    """Get topic summary from DB for --info display."""
    session = get_session(engine)
    rows = session.execute(
        __import__("sqlalchemy").text(
            "SELECT predicted_value, COUNT(*), AVG(confidence) "
            "FROM predictions WHERE label_type='topic' AND model_name='bertopic_v1' "
            "GROUP BY predicted_value ORDER BY COUNT(*) DESC"
        )
    ).fetchall()
    session.close()
    return rows


def run_topic_modeling(engine=None, refit=False):
    """Full topic modeling pipeline: load embeddings, aggregate, fit, store."""
    if engine is None:
        engine = get_local_engine()
        init_local_db(engine)

    # Check for existing assignments
    if not refit:
        session = get_session(engine)
        existing = session.execute(
            __import__("sqlalchemy").text(
                "SELECT COUNT(*) FROM predictions WHERE label_type='topic'"
            )
        ).fetchone()[0]
        session.close()
        if existing > 0:
            logger.info(f"Topic assignments already exist ({existing}). Use --refit to redo.")
            return

    # Load FAISS index and chunk_map
    logger.info("Loading FAISS index and chunk map...")
    index, chunk_map = load_index()
    if index is None:
        logger.error("No FAISS index found. Run pipeline.py first.")
        return

    # Extract vectors from FAISS
    n = index.ntotal
    dim = index.d
    vectors = np.zeros((n, dim), dtype=np.float32)
    for i in range(n):
        vectors[i] = index.reconstruct(i)

    # Aggregate to opinion level
    logger.info("Aggregating chunk embeddings to opinion level...")
    opinion_ids, embeddings, metadata = aggregate_opinion_embeddings(chunk_map, vectors)
    logger.info(f"Aggregated {n} chunks into {len(opinion_ids)} opinion embeddings")

    # Get opinion texts for c-TF-IDF (first chunk text per opinion)
    opinion_texts = {}
    for entry in chunk_map:
        oid = entry["opinion_id"]
        if oid not in opinion_texts:
            opinion_texts[oid] = entry.get("text", "")
    docs = [opinion_texts.get(oid, "") for oid in opinion_ids]

    # Fit BERTopic
    logger.info("Fitting BERTopic model...")
    topics, probs, topic_info = fit_topics(opinion_ids, embeddings, docs)
    n_topics = len(set(t for t in topics if t != -1))
    n_outliers = sum(1 for t in topics if t == -1)
    logger.info(f"Found {n_topics} topics, {n_outliers} outliers")

    # Store assignments
    store_topic_assignments(engine, opinion_ids, topics, probs)

    # Save 2D coordinates for scatter plot
    logger.info("Computing 2D UMAP projection for visualization...")
    save_2d_coords(opinion_ids, embeddings)

    # Save model record
    session = get_session(engine)
    existing_model = session.query(ModelRecord).filter_by(name="bertopic_v1").first()
    params = {
        "n_topics": n_topics,
        "n_outliers": n_outliers,
        "n_opinions": len(opinion_ids),
        "top_topics": topic_info.head(10).to_dict("records") if topic_info is not None else [],
    }
    if existing_model:
        existing_model.trained_at = datetime.now(timezone.utc).isoformat()
        existing_model.params_json = json.dumps(params, default=str)
    else:
        session.add(ModelRecord(
            name="bertopic_v1",
            label_type="topic",
            trained_at=datetime.now(timezone.utc).isoformat(),
            params_json=json.dumps(params, default=str),
        ))
    session.commit()
    session.close()

    logger.info("Topic modeling complete.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Topic modeling with BERTopic")
    parser.add_argument("--refit", action="store_true", help="Clear and refit topics")
    parser.add_argument("--info", action="store_true", help="Print topic summary")
    args = parser.parse_args()

    engine = get_local_engine()
    init_local_db(engine)

    if args.info:
        rows = get_topic_summary(engine)
        if not rows:
            print("No topic assignments found. Run: python topics.py")
        else:
            print(f"\n{'Topic':<15} {'Count':>8} {'Avg Conf':>10}")
            print("-" * 35)
            for topic, count, avg_conf in rows:
                print(f"{topic:<15} {count:>8} {avg_conf:>10.3f}")
    else:
        run_topic_modeling(engine, refit=args.refit)
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_topics.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add topics.py tests/test_topics.py
git commit -m "feat: add topic modeling pipeline with BERTopic"
```

---

### Task 3: Run topic modeling on real data

**Files:**
- None (execution only)

**Step 1: Run topic modeling**

Run: `python topics.py`
Expected output:
- "Loading FAISS index and chunk map..."
- "Aggregated 433441 chunks into ~30609 opinion embeddings"
- "Fitting BERTopic model..." (may take 2-5 min on CPU)
- "Found N topics, M outliers"
- "Stored 30609 topic assignments"
- "Saved 2D coordinates"

**Step 2: Verify results**

Run: `python topics.py --info`
Expected: Table showing topics with counts and confidence scores.

---

### Task 4: Topics dashboard page

**Files:**
- Modify: `pages/2_Topics.py`

**Step 1: Replace placeholder with full dashboard**

Replace `pages/2_Topics.py` with:

```python
import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import text

from db import get_local_engine, init_local_db, get_session

st.set_page_config(page_title="Topics", page_icon="⚖️", layout="wide")
st.title("Topic Modeling")


@st.cache_resource
def get_db_engine():
    engine = get_local_engine()
    init_local_db(engine)
    return engine


@st.cache_data
def load_topic_data(_engine):
    """Load all topic assignments with opinion metadata."""
    session = get_session(_engine)
    rows = session.execute(text(
        "SELECT p.opinion_id, p.predicted_value, p.confidence, "
        "o.title, o.court_type, o.circuit, o.date_issued "
        "FROM predictions p JOIN opinions o ON p.opinion_id = o.id "
        "WHERE p.label_type = 'topic' AND p.model_name = 'bertopic_v1'"
    )).fetchall()
    session.close()
    if not rows:
        return None
    return pd.DataFrame(rows, columns=[
        "opinion_id", "topic", "confidence",
        "title", "court_type", "circuit", "date_issued",
    ])


@st.cache_data
def load_topic_words(_engine):
    """Load topic top words from model params."""
    session = get_session(_engine)
    row = session.execute(text(
        "SELECT params_json FROM models WHERE name = 'bertopic_v1'"
    )).fetchone()
    session.close()
    if row and row[0]:
        import json
        params = json.loads(row[0])
        return params.get("top_topics", [])
    return []


@st.cache_data
def load_2d_coords():
    """Load precomputed 2D UMAP coordinates."""
    import config
    path = os.path.join(config.PROJECT_ROOT, "data", "models", "topic_coords_v1.npz")
    if not os.path.exists(path):
        return None, None
    data = np.load(path)
    return data["opinion_ids"], data["coords"]


engine = get_db_engine()
df = load_topic_data(engine)

if df is None:
    st.warning("No topic assignments found. Run `python topics.py` to fit the topic model.")
    st.stop()

# --- Sidebar Filters ---
with st.sidebar:
    st.header("Filters")

    # Statute filter via opinion_statutes join
    session = get_session(engine)
    statute_options = ["All", "FDCPA", "TCPA", "FCRA"]
    selected_statute = st.selectbox("Statute", statute_options)

    if selected_statute != "All":
        statute_ids = session.execute(text(
            "SELECT os.opinion_id FROM opinion_statutes os "
            "JOIN statutes s ON os.statute_id = s.id "
            f"WHERE UPPER(s.key) = '{selected_statute.upper()}'"
        )).fetchall()
        statute_opinion_ids = {r[0] for r in statute_ids}
        df = df[df["opinion_id"].isin(statute_opinion_ids)]

    circuits = sorted(df["circuit"].dropna().unique())
    circuits = [c for c in circuits if c]
    selected_circuit = st.selectbox("Circuit", ["All"] + circuits)
    if selected_circuit != "All":
        df = df[df["circuit"] == selected_circuit]

    court_types = ["All", "district", "circuit"]
    selected_court_type = st.selectbox("Court Type", court_types)
    if selected_court_type != "All":
        df = df[df["court_type"] == selected_court_type]

    session.close()

# Filter out outlier topic for main displays
df_no_outliers = df[df["topic"] != "topic_-1"]

# --- Summary Metrics ---
n_topics = df_no_outliers["topic"].nunique()
n_outliers = len(df[df["topic"] == "topic_-1"])
n_total = len(df)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Topics Found", n_topics)
with col2:
    st.metric("Opinions Assigned", f"{n_total - n_outliers:,}")
with col3:
    st.metric("Outliers", f"{n_outliers:,}")

# --- Topic Overview Table ---
st.subheader("Topic Overview")

topic_words = load_topic_words(engine)
word_lookup = {}
for tw in topic_words:
    tid = tw.get("Topic")
    name = tw.get("Name", "")
    if tid is not None:
        word_lookup[f"topic_{tid}"] = name

topic_counts = df_no_outliers["topic"].value_counts().reset_index()
topic_counts.columns = ["Topic", "Count"]
topic_counts["Percentage"] = (topic_counts["Count"] / n_total * 100).round(1)
topic_counts["Top Words"] = topic_counts["Topic"].map(
    lambda t: word_lookup.get(t, "")
)
topic_counts = topic_counts.sort_values("Count", ascending=False).reset_index(drop=True)

st.dataframe(topic_counts, use_container_width=True, hide_index=True)

# --- Topic Size Bar Chart ---
st.subheader("Topic Sizes")

top_20 = topic_counts.head(20)
fig_bar = px.bar(
    top_20, x="Count", y="Topic", orientation="h",
    title="Top 20 Topics by Number of Opinions",
    labels={"Count": "Number of Opinions", "Topic": ""},
)
fig_bar.update_layout(yaxis=dict(autorange="reversed"), height=600)
st.plotly_chart(fig_bar, use_container_width=True)

# --- Interactive Scatter Plot ---
st.subheader("Topic Landscape (2D UMAP)")

coord_ids, coords = load_2d_coords()
if coords is not None:
    # Build scatter dataframe by matching opinion_ids
    coord_df = pd.DataFrame({
        "opinion_id": coord_ids,
        "x": coords[:, 0],
        "y": coords[:, 1],
    })
    scatter_df = coord_df.merge(df[["opinion_id", "topic", "title"]], on="opinion_id", how="inner")

    # Limit to non-outliers for color, show outliers in gray
    scatter_df["display_topic"] = scatter_df["topic"].apply(
        lambda t: "outlier" if t == "topic_-1" else t
    )

    fig_scatter = px.scatter(
        scatter_df, x="x", y="y", color="display_topic",
        hover_data={"title": True, "topic": True, "x": False, "y": False},
        title="Opinions in Topic Space",
        labels={"x": "", "y": ""},
        opacity=0.5,
    )
    fig_scatter.update_layout(
        height=700,
        legend_title_text="Topic",
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
else:
    st.info("2D coordinates not found. Run `python topics.py` to generate them.")

# --- Topic Evolution Over Time ---
st.subheader("Topic Evolution Over Time")

df_time = df_no_outliers.copy()
df_time["year"] = pd.to_datetime(df_time["date_issued"], errors="coerce").dt.year
df_time = df_time.dropna(subset=["year"])
df_time["year"] = df_time["year"].astype(int)

if not df_time.empty:
    # Show top 10 topics only
    top_topics = df_time["topic"].value_counts().head(10).index.tolist()
    df_top = df_time[df_time["topic"].isin(top_topics)]

    evolution = df_top.groupby(["year", "topic"]).size().reset_index(name="count")

    fig_evo = px.line(
        evolution, x="year", y="count", color="topic",
        title="Topic Prevalence Over Time (Top 10 Topics)",
        labels={"year": "Year", "count": "Number of Opinions", "topic": "Topic"},
    )
    fig_evo.update_layout(height=500)
    st.plotly_chart(fig_evo, use_container_width=True)
else:
    st.info("No date information available for temporal analysis.")
```

**Step 2: Verify the page loads**

Run: `python -m streamlit run app.py --server.headless true`
Navigate to Topics page and verify all four sections render.

**Step 3: Commit**

```bash
git add pages/2_Topics.py
git commit -m "feat: build interactive Topics dashboard with scatter plot and temporal evolution"
```

---

### Task 5: Pipeline integration

**Files:**
- Modify: `pipeline.py:24,31,209-217`

**Step 1: Add import and --topics flag**

In `pipeline.py`:

At line 24 (imports), add:
```python
from topics import run_topic_modeling
```

In `run_pipeline` function signature (line 31), add `topics=False` parameter:
```python
def run_pipeline(sync_only=False, reindex=False, classify=False, predict_new=False, topics=False):
```

After the `predict_new` block (after line 192), add:
```python
    if topics:
        logger.info("Running topic modeling...")
        run_topic_modeling(engine, refit=reindex)
```

In the argparse section, add:
```python
    parser.add_argument("--topics", action="store_true", help="Run topic modeling after indexing")
```

In the `run_pipeline` call, add:
```python
        topics=args.topics,
```

**Step 2: Verify pipeline runs with --topics flag**

Run: `python pipeline.py --topics`
Expected: Pipeline runs sync, then topic modeling (should say "Topic assignments already exist" since we ran it in Task 3).

**Step 3: Commit**

```bash
git add pipeline.py
git commit -m "feat: add --topics flag to pipeline for topic modeling"
```

---

### Task 6: Run all tests and push

**Files:**
- None (validation only)

**Step 1: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All tests pass (64 existing + 6 new = 70 tests).

**Step 2: Commit any fixes if needed**

**Step 3: Push**

```bash
git push
```
