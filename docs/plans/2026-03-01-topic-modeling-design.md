# Topic Modeling — Design Document

## Goal

Discover natural topics across 30K federal court opinions using BERTopic, store assignments, and visualize in an interactive Topics dashboard with filtering and temporal analysis.

## Current State

- 30,609 opinions with text across FDCPA, TCPA, FCRA
- 433,441 chunk embeddings in FAISS (all-MiniLM-L6-v2, 384-dim)
- Placeholder Topics page (`pages/2_Topics.py`)
- Analytics dashboard as UI pattern reference

## Approach

### Embedding Aggregation

Work at the opinion level (30K vectors, not 433K chunks). Aggregate chunk embeddings by averaging all chunks per opinion, then L2-normalize. This produces one 384-dim vector per opinion that captures the full document's semantics.

### BERTopic Pipeline

- Pass pre-computed opinion embeddings to BERTopic (skips internal embedding)
- UMAP reduces 384→5D for clustering
- HDBSCAN auto-discovers topic count (expect 30-80 topics)
- c-TF-IDF extracts representative words per topic
- Outlier opinions get topic -1

### Storage

- Topic assignments stored in `predictions` table: one row per opinion, `label_type="topic"`, `predicted_value="topic_N"`, confidence from HDBSCAN probability
- Topic metadata (top words, sizes) stored in model's `params_json`
- 2D UMAP coordinates saved to `data/models/topic_coords_v1.npz` for scatter plot
- Fitted BERTopic model saved to `data/models/bertopic_v1.pkl`

### Topics Page UI

Sidebar filters: statute, circuit, court type (same pattern as Analytics).

Four visualizations:
1. **Topic Overview Table** — Each topic's ID, top 5 words, opinion count, corpus percentage. Outliers shown at bottom.
2. **Topic Size Bar Chart** — Horizontal bars of topic sizes (excluding outliers).
3. **Interactive Scatter Plot** — 2D UMAP projection colored by topic, Plotly hover shows title and topic words. Filterable.
4. **Topic Evolution Over Time** — Line chart of topic prevalence by year for top 10 topics.

### Pipeline Integration

- `python topics.py` — fit model + store assignments
- `python topics.py --refit` — clear old assignments, refit
- `python topics.py --info` — print topic summary
- `pipeline.py --topics` flag to run after indexing

## New/Modified Files

| File | Change |
|------|--------|
| `topics.py` | New: embedding aggregation, BERTopic fitting, storage, CLI |
| `pages/2_Topics.py` | Replace placeholder with dashboard |
| `pipeline.py` | Add `--topics` flag |
| `requirements.txt` | Add bertopic, umap-learn, hdbscan, plotly |
| `tests/test_topics.py` | Tests for aggregation, fitting, storage |

## Dependencies

- `bertopic>=0.16.0`
- `umap-learn>=0.5.0`
- `hdbscan>=0.8.33`
- `plotly>=5.0.0`

All verified compatible with Python 3.14.
