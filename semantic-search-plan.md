# Semantic Search for Federal Opinions — Implementation Plan

## Overview

Build a local semantic search system over ~30,000 federal court opinions (FDCPA, TCPA, FCRA) stored in plaintext. The system converts opinions into vector embeddings, stores them in a local vector database, and provides a web UI for natural-language search with metadata filtering.

**All processing is local. No data leaves the machine.**

---

## Prerequisites

- Python 3.10+
- 8GB+ RAM (16GB recommended for embedding 30K docs)
- ~5GB free disk space (vectors + model weights)
- Access to the existing database containing the 30K opinions in plaintext

---

## Project Structure

```
federal-opinion-search/
├── config.py                # DB connection strings, model settings, chunk params
├── extract.py               # Pull opinions from existing database
├── chunk.py                 # Split opinions into searchable segments
├── embed.py                 # Generate vector embeddings
├── index.py                 # Build and manage ChromaDB collection
├── search.py                # Search interface (CLI)
├── app.py                   # Streamlit web UI
├── requirements.txt
└── data/
    └── chroma_db/           # Persistent vector store
```

---

## Phase 1: Data Extraction

**Goal:** Pull all opinions from the existing database into a standardized format.

### Tasks

1. Connect to the existing database (get connection details from user — likely SQLite or PostgreSQL).
2. Extract each opinion with all available metadata:
   - `opinion_id` (primary key or unique identifier)
   - `case_name`
   - `citation` (if available)
   - `court` (district, circuit)
   - `date_decided`
   - `statute` (FDCPA, TCPA, FCRA — whichever applies)
   - `full_text` (the plaintext opinion)
   - Any other fields present (judge, docket number, etc.)
3. Write to a standardized list of dictionaries or a staging table.
4. Log: total count, any opinions with missing text, character length distribution.

### Output
A Python list or generator of opinion dicts, each with metadata + full_text.

---

## Phase 2: Chunking

**Goal:** Split each opinion into overlapping segments suitable for embedding.

### Strategy

- **Chunk size:** ~500 tokens (~2,000 characters). This balances context vs. embedding model token limits.
- **Overlap:** 100 tokens (~400 characters) between chunks. Prevents losing context at boundaries.
- **Method:** Split on paragraph boundaries first. If a paragraph exceeds chunk size, split on sentence boundaries. Never split mid-sentence.
- **Preserve metadata:** Each chunk inherits the parent opinion's metadata plus:
  - `chunk_index` (position within the opinion, 0-indexed)
  - `chunk_id` (unique: `{opinion_id}_chunk_{chunk_index}`)

### Tasks

1. Implement `chunk_opinion(opinion_dict) -> list[chunk_dicts]`.
2. Use `nltk.sent_tokenize` or regex for sentence splitting.
3. Process all 30K opinions. Expect 200K–500K total chunks.
4. Log: total chunks, avg chunks per opinion, any opinions that produced 0 chunks (empty text).

### Edge Cases
- Very short opinions (< 1 chunk): keep as a single chunk, no overlap.
- Extremely long opinions: just produce more chunks; no special handling needed.

---

## Phase 3: Embedding

**Goal:** Convert each text chunk into a dense vector.

### Model Selection

Use `sentence-transformers/all-MiniLM-L6-v2` as the default:
- 384-dimensional vectors
- ~80MB model
- Fast on CPU (~1,000 chunks/minute)
- Good general-purpose quality

If quality is insufficient after testing, upgrade to `all-mpnet-base-v2` (768-dim, ~420MB, slower but better).

### Tasks

1. Install: `pip install sentence-transformers`
2. Load model: `SentenceTransformer('all-MiniLM-L6-v2')`
3. Encode all chunks in batches:
   ```python
   embeddings = model.encode(
       texts,
       batch_size=64,
       show_progress_bar=True,
       normalize_embeddings=True  # enables cosine similarity via dot product
   )
   ```
4. Expect ~2-6 hours for 300K+ chunks on CPU. GPU cuts this to ~15-30 minutes.
5. Save embeddings to disk as a numpy array checkpoint before loading into ChromaDB (safety net).

### Output
A numpy array of shape `(num_chunks, 384)` plus the corresponding chunk metadata.

---

## Phase 4: Vector Database (ChromaDB)

**Goal:** Store embeddings + metadata in a persistent, queryable vector database.

### Why ChromaDB
- Pure Python, no external services
- Persistent storage to disk
- Built-in metadata filtering
- Simple API

### Tasks

1. Install: `pip install chromadb`
2. Create persistent client and collection:
   ```python
   import chromadb
   client = chromadb.PersistentClient(path="./data/chroma_db")
   collection = client.get_or_create_collection(
       name="federal_opinions",
       metadata={"hnsw:space": "cosine"}
   )
   ```
3. Add chunks in batches (ChromaDB has a batch limit of ~5,000 per `.add()` call):
   ```python
   for i in range(0, len(chunks), 5000):
       batch = chunks[i:i+5000]
       collection.add(
           ids=[c['chunk_id'] for c in batch],
           embeddings=embeddings[i:i+5000].tolist(),
           documents=[c['text'] for c in batch],
           metadatas=[{
               'opinion_id': c['opinion_id'],
               'case_name': c['case_name'],
               'court': c['court'],
               'date': c['date'],
               'statute': c['statute'],
               'chunk_index': c['chunk_index']
           } for c in batch]
       )
   ```
4. Verify: `collection.count()` should match total chunks.

### Updating
To add new opinions later, just chunk → embed → add. ChromaDB handles deduplication by ID.

---

## Phase 5: Search Engine

**Goal:** Accept natural-language queries and return ranked results with metadata.

### Core Search Function

```python
def search(query: str, n_results: int = 10, filters: dict = None):
    where_clause = {}
    if filters:
        if filters.get('statute'):
            where_clause['statute'] = filters['statute']
        if filters.get('court'):
            where_clause['court'] = filters['court']
        # Date range filtering uses $gte/$lte operators
    
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_clause if where_clause else None,
        include=["documents", "metadatas", "distances"]
    )
    return results
```

### Result Grouping

Multiple chunks from the same opinion may appear in results. Group them:

1. Query with `n_results=50` (oversample).
2. Group chunks by `opinion_id`.
3. For each opinion, take the best (lowest distance) chunk as the representative.
4. Sort opinions by their best chunk's score.
5. Return top N opinions, each with its best-matching passage highlighted.

### CLI Interface

Build `search.py` as a quick CLI tool:
```
python search.py "collector called third party and disclosed debt" --statute FDCPA --court "5th Circuit" --top 10
```

---

## Phase 6: Web UI (Streamlit)

**Goal:** A browser-based search interface.

### Layout

1. **Sidebar:**
   - Statute filter (multiselect: FDCPA, TCPA, FCRA)
   - Court filter (multiselect, populated from metadata)
   - Date range slider
   - Results per page slider (5-50)

2. **Main area:**
   - Search box (large, top of page)
   - Results list, each showing:
     - Case name (bold) + citation
     - Court | Date | Statute
     - Best-matching passage (the chunk text) with the query terms contextually highlighted
     - Similarity score (as a percentage or bar)
     - Expandable section to show full opinion text
   - "Load more" or pagination

### Tasks

1. Install: `pip install streamlit`
2. Build `app.py` with the layout above.
3. Cache the ChromaDB collection and embedding model using `@st.cache_resource`.
4. Run: `streamlit run app.py`

---

## Phase 7: Enhancements (Optional, In Priority Order)

### 7a. Cross-Encoder Re-ranking

After initial vector search returns top 50, re-rank with a cross-encoder for better precision:

```python
from sentence_transformers import CrossEncoder
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
scores = reranker.predict([(query, doc) for doc in candidate_docs])
```

This is slower but significantly improves result quality for the final top 10.

### 7b. Hybrid Search (Semantic + Keyword)

Add BM25 keyword search alongside semantic search:

```python
pip install rank-bm25
```

Combine scores: `final_score = 0.7 * semantic_score + 0.3 * bm25_score`

This catches cases where exact statutory language (like "1692e(8)") matters.

### 7c. Named Entity Extraction

Run spaCy NER over opinions to extract structured fields:
- Judge names
- Party names (plaintiff/defendant)
- Dollar amounts (damages awarded)
- Statutory sections cited

Store as additional metadata in ChromaDB for filtering.

### 7d. Full-Text Retrieval

Add a "view full opinion" button that pulls the complete text from the source database, so users can read the whole opinion after finding it via search.

---

## Dependencies (requirements.txt)

```
sentence-transformers>=2.2.0
chromadb>=0.4.0
streamlit>=1.28.0
nltk>=3.8
numpy>=1.24.0
# Optional enhancements:
# rank-bm25>=0.2.2
# spacy>=3.7.0
```

---

## Implementation Order

| Step | Phase | Estimated Time | Description |
|------|-------|---------------|-------------|
| 1 | Phase 1 | 1 hour | Connect to DB, extract opinions, validate data |
| 2 | Phase 2 | 1-2 hours | Implement chunking, process all opinions |
| 3 | Phase 3 | 1 hour code + 2-6 hours compute | Embed all chunks, save checkpoint |
| 4 | Phase 4 | 1 hour | Load into ChromaDB, verify counts |
| 5 | Phase 5 | 1-2 hours | Build search function with grouping + CLI |
| 6 | Phase 6 | 2-3 hours | Streamlit UI with filters and display |
| 7 | Phase 7 | As needed | Re-ranking, hybrid search, NER |

**Total active coding time: ~8-12 hours**
**Total wall-clock time (including embedding compute): ~1-2 days**

---

## Notes for Claude Code

- Ask the user for their database connection details and schema before starting Phase 1.
- The user's database likely has specific column names — adapt `extract.py` accordingly.
- Test each phase independently before moving to the next.
- After Phase 4, run a few sample queries to verify the system works before building the UI.
- The user runs a privacy-first setup — confirm everything stays local, no API calls to external embedding services.
- The opinions span three statutes and date back to 1977, so date filtering will be important.
- The user may want to integrate this with their existing AI litigation toolbox later — keep the search module importable (not just a script).
