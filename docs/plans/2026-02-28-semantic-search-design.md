# Semantic Search System — Design Document

## Overview

A local semantic search engine over ~30,000 federal court opinions (FDCPA, TCPA, FCRA) sourced from a SQLite database in the `cld` project. Opinions are synced to a local database, chunked, embedded, stored in ChromaDB, and served through a Streamlit web UI with metadata filtering.

All processing is local. No data leaves the machine.

---

## Architecture

Three layers:

1. **Data layer** — Sync script pulls opinions from `cld/fdcpa.db` into a local `data/opinions.db`. One-directional, never writes back.
2. **Search layer** — Opinions are chunked, embedded with sentence-transformers, and indexed in ChromaDB. Search queries are embedded with the same model and matched against the index.
3. **UI layer** — Streamlit multi-page app. Search page is fully built; Topics, Analytics, and Citations pages are scaffolded as placeholders.

---

## Project Structure

```
ml/
├── config.py              # paths, model name, chunk params
├── db.py                  # local SQLite helpers
├── sync.py                # pull from cld/fdcpa.db → local DB
├── chunk.py               # opinion → chunks
├── embed.py               # chunks → vectors
├── index.py               # load vectors into ChromaDB
├── search.py              # query interface + CLI
├── app.py                 # Streamlit entry point
├── pages/
│   ├── 1_Search.py        # full search UI
│   ├── 2_Topics.py        # placeholder
│   ├── 3_Analytics.py     # placeholder
│   └── 4_Citations.py     # placeholder
├── requirements.txt
├── docs/
│   └── plans/
└── data/
    ├── opinions.db        # local synced copy
    ├── chroma_db/         # vector store
    └── checkpoints/       # embedding checkpoints
```

---

## Data Sync (`sync.py`)

**Source:** `C:/PythonProject/cld/fdcpa.db` (configurable in `config.py`)

**Local SQLite schema:**

```sql
statutes
├── id (PK)
├── key (TEXT, unique — "fdcpa", "tcpa", "fcra")
└── name (TEXT)

opinions
├── id (PK, matches cld opinion id)
├── package_id (TEXT)
├── title (TEXT)
├── court_name (TEXT)
├── court_type (TEXT — "district", "circuit")
├── circuit (TEXT — "5th", "9th", etc.)
├── date_issued (TEXT, ISO date)
├── plain_text (TEXT)
├── pdf_url (TEXT)
├── synced_at (TEXT, ISO timestamp)
└── chunked (INTEGER, 0 or 1)

opinion_statutes
├── opinion_id (FK → opinions.id)
└── statute_id (FK → statutes.id)

fdcpa_sections
├── opinion_id (FK → opinions.id)
├── subsection (TEXT — "1692d", "1692e", etc.)
└── description (TEXT)
```

**Sync behavior:**
- First run: copies all opinions that have `plain_text`, plus statutes, opinion_statutes, and fdcpa_sections.
- Subsequent runs: pulls only opinions added or modified since last sync (compares `last_modified`/`created_at`). New opinions are marked `chunked=0`.
- If source DB is locked (scraper running), retry once after 5 seconds, then exit with a clear message.

---

## Chunking (`chunk.py`)

**Parameters:**
- Chunk size: ~500 tokens (~2,000 characters)
- Overlap: ~100 tokens (~400 characters)
- Split hierarchy: paragraph boundaries → sentence boundaries → never mid-sentence
- Sentence tokenization: `nltk.sent_tokenize`

**Each chunk carries:**
- `chunk_id`: `"{opinion_id}_chunk_{chunk_index}"`
- `opinion_id`: parent opinion FK
- `chunk_index`: position within opinion (0-indexed)
- `text`: chunk content

**Edge cases:**
- No text or whitespace only: skip, log warning
- Short opinions (< 500 tokens): single chunk, no overlap
- Very long opinions: more chunks, no special handling

---

## Embedding (`embed.py`)

**Model (configurable):**
- Default: `all-MiniLM-L6-v2` (384-dim, ~80MB, ~1,000 chunks/min on CPU)
- Upgrade path: `all-mpnet-base-v2` (768-dim, better quality)
- Model name stored in `config.py` and in ChromaDB collection metadata

**Process:**
- Encode in batches of 64 with `normalize_embeddings=True`
- Checkpoint to `data/checkpoints/` after every 500 opinions
- On crash, resume from last checkpoint
- Estimated time for full corpus (~300K chunks): 2-3 hours on CPU

**Model change detection:**
- ChromaDB collection metadata stores the model name used for indexing
- If `config.EMBEDDING_MODEL` differs from stored model, refuse to add and prompt for full re-index

---

## ChromaDB Indexing (`index.py`)

**Collection:**
```python
collection = client.get_or_create_collection(
    name="federal_opinions",
    metadata={"hnsw:space": "cosine", "embedding_model": EMBEDDING_MODEL}
)
```

**Chunk metadata stored in ChromaDB:**
- `opinion_id` (int)
- `title` (string)
- `court_name` (string)
- `court_type` (string — "district" or "circuit")
- `circuit` (string)
- `date_issued` (ISO date string)
- `statutes` (comma-separated string, e.g., "FDCPA,FCRA")
- `chunk_index` (int)

Batch inserts in groups of 5,000 (ChromaDB limit).

**Incremental updates:** Pipeline only processes `chunked=0` opinions. ChromaDB deduplicates by `chunk_id`.

---

## Search Engine (`search.py`)

**Flow:**
1. Embed query with the same model
2. Query ChromaDB for top 50 chunks (oversample) with filters
3. Group chunks by `opinion_id`
4. Score each opinion by its best chunk (lowest distance)
5. Return top N opinions with best passage and metadata

**Filtering:**
- Statute: `{"statutes": {"$contains": "FDCPA"}}`
- Circuit: `{"circuit": "5th"}`
- Court type: `{"court_type": "district"}`
- Date range: `{"$and": [{"date_issued": {"$gte": "..."}}, {"date_issued": {"$lte": "..."}}]}`
- FDCPA subsection: two-step — query SQLite for matching opinion IDs, then pass as `{"opinion_id": {"$in": [...]}}` to ChromaDB. If ID list is very large, filter post-retrieval instead.

**Result object per opinion:**
```
opinion_id, title, court_name, circuit, date_issued,
statutes, similarity_score, best_passage, chunk_index
```

**CLI:**
```
python search.py "collector called third party" --statute FDCPA --circuit "5th" --top 10
```

---

## Streamlit UI

**Entry point:** `app.py` — title, nav, cached resources.

### Search Page (`pages/1_Search.py`)

**Sidebar filters:**
- Statute multiselect (FDCPA, TCPA, FCRA)
- Circuit multiselect (populated from DB)
- Court type radio (All / District / Circuit)
- Date range slider (1983–2026)
- FDCPA subsection multiselect (visible only when FDCPA selected, populated from `fdcpa_sections`)
- Results per page slider (5–50, default 10)

**Main area:**
- Large search box at top
- Results list:
  - Case name (bold, clickable to detail page) + statute badges
  - Court | Circuit | Date
  - Best-matching passage with similarity score bar
  - "Show more" expandable for full opinion text inline
- Pagination at bottom

**Detail page:**
- Full metadata header
- All matching passages highlighted within full opinion text
- Statutes and FDCPA subsections listed

### Placeholder Pages
- `2_Topics.py`, `3_Analytics.py`, `4_Citations.py` — title + "Coming soon" message each

**Performance:**
- ChromaDB collection and embedding model cached with `@st.cache_resource`
- SQLite connection cached similarly

---

## Configuration (`config.py`)

```python
# Paths
SOURCE_DB = "C:/PythonProject/cld/fdcpa.db"
LOCAL_DB = "data/opinions.db"
CHROMA_DIR = "data/chroma_db"
CHECKPOINT_DIR = "data/checkpoints"

# Embedding
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Chunking
CHUNK_SIZE = 2000        # characters (~500 tokens)
CHUNK_OVERLAP = 400      # characters (~100 tokens)

# Search
DEFAULT_TOP_K = 10
OVERSAMPLE_FACTOR = 5    # fetch 5x results for grouping
```

---

## Error Handling

- **Sync:** Retry once after 5 seconds if source DB locked, then exit with message.
- **Embedding:** Checkpoint after every 500 opinions. Resume from last checkpoint on crash.
- **Model mismatch:** Refuse to mix vectors from different models. Prompt for full re-index.
- **Empty index:** Show friendly "No opinions indexed. Run the pipeline first." in UI.

## Logging

- Python `logging` module (not print)
- Output to console + `data/pipeline.log`
- INFO for progress, WARNING for skipped opinions, ERROR for failures

---

## Implementation Order

| Step | Module | Description |
|------|--------|-------------|
| 1 | `config.py`, `db.py` | Configuration and local DB setup |
| 2 | `sync.py` | Pull opinions from cld → local DB |
| 3 | `chunk.py` | Chunking pipeline |
| 4 | `embed.py`, `index.py` | Embedding + ChromaDB indexing |
| 5 | `search.py` | Search engine + CLI |
| 6 | `app.py`, `pages/` | Streamlit UI |

---

## Dependencies

```
sentence-transformers>=2.2.0
chromadb>=0.4.0
streamlit>=1.28.0
nltk>=3.8
numpy>=1.24.0
sqlalchemy>=2.0.0
```

---

## Future Enhancements (not in scope now)

- Cross-encoder re-ranking for better precision
- Hybrid search (semantic + BM25 keyword)
- Named entity extraction for richer metadata
- Full-text highlighting with exact query term matching
