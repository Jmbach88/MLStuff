# Semantic Search Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a local semantic search engine over ~30K federal court opinions with a Streamlit UI.

**Architecture:** Sync opinions from `cld/fdcpa.db` into a local SQLite DB, chunk them, embed with sentence-transformers, index in ChromaDB, and serve through a Streamlit multi-page app with metadata filtering. All local, no external API calls.

**Tech Stack:** Python 3.10+, sentence-transformers (all-MiniLM-L6-v2), ChromaDB, Streamlit, SQLAlchemy, NLTK, NumPy

---

### Task 1: Project Setup — Config and Requirements

**Files:**
- Create: `config.py`
- Create: `requirements.txt`

**Step 1: Create `requirements.txt`**

```
sentence-transformers>=2.2.0
chromadb>=0.4.0
streamlit>=1.28.0
nltk>=3.8
numpy>=1.24.0
sqlalchemy>=2.0.0
```

**Step 2: Create `config.py`**

```python
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
SOURCE_DB = "C:/PythonProject/cld/fdcpa.db"
LOCAL_DB = str(PROJECT_ROOT / "data" / "opinions.db")
CHROMA_DIR = str(PROJECT_ROOT / "data" / "chroma_db")
CHECKPOINT_DIR = str(PROJECT_ROOT / "data" / "checkpoints")

# Embedding
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Chunking
CHUNK_SIZE = 2000       # characters (~500 tokens)
CHUNK_OVERLAP = 400     # characters (~100 tokens)

# Search
DEFAULT_TOP_K = 10
OVERSAMPLE_FACTOR = 5   # fetch 5x results for grouping
```

**Step 3: Create data directories and install dependencies**

Run:
```bash
mkdir -p data/chroma_db data/checkpoints
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt_tab')"
```

**Step 4: Create `.gitignore`**

```
data/
__pycache__/
*.pyc
.env
```

**Step 5: Commit**

```bash
git add config.py requirements.txt .gitignore
git commit -m "feat: project setup with config and dependencies"
```

---

### Task 2: Local Database Schema (`db.py`)

**Files:**
- Create: `db.py`
- Create: `tests/test_db.py`

**Step 1: Write the failing test**

```python
# tests/test_db.py
import os
import pytest
from sqlalchemy import inspect

# Use a temp DB for tests
os.environ["ML_LOCAL_DB"] = ":memory:"


def test_init_db_creates_all_tables():
    from db import init_local_db, get_local_engine

    engine = get_local_engine()
    init_local_db(engine)
    table_names = inspect(engine).get_table_names()
    assert "opinions" in table_names
    assert "statutes" in table_names
    assert "opinion_statutes" in table_names
    assert "fdcpa_sections" in table_names


def test_opinion_columns_exist():
    from db import init_local_db, get_local_engine

    engine = get_local_engine()
    init_local_db(engine)
    cols = [c["name"] for c in inspect(engine).get_columns("opinions")]
    for expected in ["id", "package_id", "title", "court_name", "court_type",
                     "circuit", "date_issued", "plain_text", "pdf_url",
                     "synced_at", "chunked"]:
        assert expected in cols, f"Missing column: {expected}"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_db.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'db'`

**Step 3: Write `db.py`**

```python
# db.py
import os
import logging
from datetime import datetime, timezone

from sqlalchemy import (
    Column, Integer, Text, ForeignKey, UniqueConstraint,
    create_engine, inspect
)
from sqlalchemy.orm import declarative_base, relationship, Session

import config

logger = logging.getLogger(__name__)

Base = declarative_base()


class Statute(Base):
    __tablename__ = "statutes"
    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(Text, unique=True, nullable=False)
    name = Column(Text, nullable=False)
    opinions = relationship("Opinion", secondary="opinion_statutes", back_populates="statutes")


class Opinion(Base):
    __tablename__ = "opinions"
    id = Column(Integer, primary_key=True)
    package_id = Column(Text, unique=True, nullable=False)
    title = Column(Text, nullable=False)
    court_name = Column(Text)
    court_type = Column(Text)
    circuit = Column(Text)
    date_issued = Column(Text)
    plain_text = Column(Text)
    pdf_url = Column(Text)
    synced_at = Column(Text)
    chunked = Column(Integer, default=0)
    statutes = relationship("Statute", secondary="opinion_statutes", back_populates="opinions")
    sections = relationship("FDCPASection", back_populates="opinion")


class OpinionStatute(Base):
    __tablename__ = "opinion_statutes"
    opinion_id = Column(Integer, ForeignKey("opinions.id"), primary_key=True)
    statute_id = Column(Integer, ForeignKey("statutes.id"), primary_key=True)


class FDCPASection(Base):
    __tablename__ = "fdcpa_sections"
    id = Column(Integer, primary_key=True, autoincrement=True)
    opinion_id = Column(Integer, ForeignKey("opinions.id"), nullable=False)
    subsection = Column(Text, nullable=False)
    description = Column(Text)
    opinion = relationship("Opinion", back_populates="sections")
    __table_args__ = (
        UniqueConstraint("opinion_id", "subsection", name="uq_opinion_subsection"),
    )


def get_local_engine():
    db_path = os.environ.get("ML_LOCAL_DB", config.LOCAL_DB)
    if db_path == ":memory:":
        return create_engine("sqlite:///:memory:")
    return create_engine(f"sqlite:///{db_path}")


def init_local_db(engine=None):
    if engine is None:
        engine = get_local_engine()
    Base.metadata.create_all(engine)
    return engine


def get_session(engine=None):
    if engine is None:
        engine = get_local_engine()
    return Session(engine)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_db.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add db.py tests/test_db.py
git commit -m "feat: local database schema with opinions, statutes, sections"
```

---

### Task 3: Sync Script (`sync.py`)

**Files:**
- Create: `sync.py`
- Create: `tests/test_sync.py`

**Step 1: Write the failing test**

```python
# tests/test_sync.py
import os
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

os.environ["ML_LOCAL_DB"] = ":memory:"

# We need a source DB fixture that mimics cld's schema
@pytest.fixture
def source_engine(tmp_path):
    """Create a minimal source DB mimicking cld/fdcpa.db schema."""
    db_path = str(tmp_path / "source.db")
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        conn.execute("""
            CREATE TABLE statutes (
                id INTEGER PRIMARY KEY,
                key TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE opinions (
                id INTEGER PRIMARY KEY,
                package_id TEXT UNIQUE NOT NULL,
                granule_id TEXT,
                title TEXT NOT NULL,
                court_name TEXT,
                court_type TEXT,
                circuit TEXT,
                date_issued TEXT,
                last_modified TEXT,
                pdf_url TEXT,
                plain_text TEXT,
                xml_url TEXT,
                govinfo_url TEXT,
                created_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE opinion_statutes (
                opinion_id INTEGER,
                statute_id INTEGER,
                alerted_at TEXT,
                citation_count INTEGER,
                relevance TEXT,
                review_status TEXT,
                reviewed_at TEXT,
                scanned_at TEXT,
                PRIMARY KEY (opinion_id, statute_id)
            )
        """)
        conn.execute("""
            CREATE TABLE fdcpa_sections (
                id INTEGER PRIMARY KEY,
                opinion_id INTEGER NOT NULL,
                subsection TEXT NOT NULL,
                description TEXT,
                UNIQUE(opinion_id, subsection)
            )
        """)
        # Insert test data
        conn.execute("""
            INSERT INTO statutes VALUES (1, 'tcpa', 'Telephone Consumer Protection Act'),
                                        (2, 'fdcpa', 'Fair Debt Collection Practices Act'),
                                        (3, 'fcra', 'Fair Credit Reporting Act')
        """)
        conn.execute("""
            INSERT INTO opinions (id, package_id, title, court_name, court_type, circuit,
                                  date_issued, plain_text, created_at)
            VALUES (1, 'PKG-001', 'Smith v. Collections Inc', 'US District Court SD Texas',
                    'district', '5th', '2024-01-15', 'This is the opinion text about FDCPA violations.',
                    '2024-01-16T00:00:00Z'),
                   (2, 'PKG-002', 'Jones v. Bank Corp', 'US Court of Appeals 11th Circuit',
                    'circuit', '11th', '2024-03-20', 'This opinion discusses TCPA and FDCPA claims.',
                    '2024-03-21T00:00:00Z'),
                   (3, 'PKG-003', 'No Text Case', 'US District Court ND Ohio',
                    'district', '6th', '2024-05-01', NULL,
                    '2024-05-02T00:00:00Z')
        """)
        conn.execute("""
            INSERT INTO opinion_statutes (opinion_id, statute_id)
            VALUES (1, 2), (2, 1), (2, 2), (3, 2)
        """)
        conn.execute("""
            INSERT INTO fdcpa_sections (opinion_id, subsection, description)
            VALUES (1, '1692e', 'False representations')
        """)
    return engine


def test_sync_copies_opinions_with_text(source_engine, tmp_path):
    from db import init_local_db, get_session, Opinion
    from sync import sync_from_source

    local_engine = init_local_db()
    source_url = str(source_engine.url)

    count = sync_from_source(source_url, local_engine)

    session = get_session(local_engine)
    opinions = session.query(Opinion).all()
    # Should only sync opinions 1 and 2 (opinion 3 has no text)
    assert len(opinions) == 2
    assert count == 2
    # All should be marked as not yet chunked
    assert all(o.chunked == 0 for o in opinions)
    session.close()


def test_sync_copies_statutes_and_links(source_engine):
    from db import init_local_db, get_session, Opinion, Statute
    from sync import sync_from_source

    local_engine = init_local_db()
    source_url = str(source_engine.url)
    sync_from_source(source_url, local_engine)

    session = get_session(local_engine)
    statutes = session.query(Statute).all()
    assert len(statutes) == 3

    # Opinion 2 should be linked to both TCPA and FDCPA
    op2 = session.query(Opinion).filter_by(package_id="PKG-002").first()
    statute_keys = sorted([s.key for s in op2.statutes])
    assert statute_keys == ["fdcpa", "tcpa"]
    session.close()


def test_sync_incremental_skips_existing(source_engine):
    from db import init_local_db
    from sync import sync_from_source

    local_engine = init_local_db()
    source_url = str(source_engine.url)

    # First sync
    count1 = sync_from_source(source_url, local_engine)
    assert count1 == 2

    # Second sync — no new opinions
    count2 = sync_from_source(source_url, local_engine)
    assert count2 == 0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sync.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sync'`

**Step 3: Write `sync.py`**

```python
# sync.py
import logging
from datetime import datetime, timezone

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

import config
from db import (
    Base, Opinion, Statute, OpinionStatute, FDCPASection,
    init_local_db, get_local_engine
)

logger = logging.getLogger(__name__)


def sync_from_source(source_url=None, local_engine=None):
    """Pull opinions from the cld source DB into the local DB.

    Returns the number of new opinions synced.
    """
    if source_url is None:
        source_url = f"sqlite:///{config.SOURCE_DB}"
    if local_engine is None:
        local_engine = get_local_engine()
        init_local_db(local_engine)

    source_engine = create_engine(source_url)
    now = datetime.now(timezone.utc).isoformat()
    new_count = 0

    local_session = Session(local_engine)
    try:
        # Sync statutes first
        _sync_statutes(source_engine, local_session)

        # Get existing package_ids to skip
        existing_ids = set(
            row[0] for row in local_session.execute(
                text("SELECT package_id FROM opinions")
            ).fetchall()
        )

        # Pull opinions with plain_text from source
        with Session(source_engine) as source_session:
            rows = source_session.execute(text("""
                SELECT id, package_id, title, court_name, court_type, circuit,
                       date_issued, plain_text, pdf_url
                FROM opinions
                WHERE plain_text IS NOT NULL AND plain_text != ''
            """)).fetchall()

            for row in rows:
                src_id, package_id, title, court_name, court_type, circuit, \
                    date_issued, plain_text, pdf_url = row

                if package_id in existing_ids:
                    continue

                opinion = Opinion(
                    id=src_id,
                    package_id=package_id,
                    title=title,
                    court_name=court_name,
                    court_type=court_type,
                    circuit=circuit,
                    date_issued=date_issued,
                    plain_text=plain_text,
                    pdf_url=pdf_url,
                    synced_at=now,
                    chunked=0,
                )
                local_session.add(opinion)
                new_count += 1

                # Sync statute links for this opinion
                links = source_session.execute(text(
                    "SELECT statute_id FROM opinion_statutes WHERE opinion_id = :oid"
                ), {"oid": src_id}).fetchall()
                for (statute_id,) in links:
                    local_session.add(OpinionStatute(
                        opinion_id=src_id, statute_id=statute_id
                    ))

                # Sync fdcpa_sections for this opinion
                sections = source_session.execute(text(
                    "SELECT subsection, description FROM fdcpa_sections WHERE opinion_id = :oid"
                ), {"oid": src_id}).fetchall()
                for subsection, description in sections:
                    local_session.add(FDCPASection(
                        opinion_id=src_id,
                        subsection=subsection,
                        description=description,
                    ))

        local_session.commit()
        logger.info(f"Synced {new_count} new opinions")

    except Exception:
        local_session.rollback()
        raise
    finally:
        local_session.close()

    return new_count


def _sync_statutes(source_engine, local_session):
    """Copy statute definitions from source if not already present."""
    existing = set(
        row[0] for row in local_session.execute(
            text("SELECT key FROM statutes")
        ).fetchall()
    )
    if existing:
        return

    with Session(source_engine) as source_session:
        rows = source_session.execute(
            text("SELECT id, key, name FROM statutes")
        ).fetchall()
        for sid, key, name in rows:
            local_session.add(Statute(id=sid, key=key, name=name))
    local_session.flush()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = get_local_engine()
    init_local_db(engine)
    count = sync_from_source(local_engine=engine)
    print(f"Synced {count} opinions")
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_sync.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add sync.py tests/test_sync.py
git commit -m "feat: sync script to pull opinions from cld source DB"
```

---

### Task 4: Chunking Pipeline (`chunk.py`)

**Files:**
- Create: `chunk.py`
- Create: `tests/test_chunk.py`

**Step 1: Write the failing test**

```python
# tests/test_chunk.py
import pytest
from chunk import chunk_opinion


def test_short_opinion_produces_single_chunk():
    """Opinion shorter than CHUNK_SIZE → one chunk."""
    text = "This is a short opinion. It has two sentences."
    chunks = chunk_opinion(opinion_id=1, text=text, chunk_size=2000, overlap=400)
    assert len(chunks) == 1
    assert chunks[0]["opinion_id"] == 1
    assert chunks[0]["chunk_id"] == "1_chunk_0"
    assert chunks[0]["chunk_index"] == 0
    assert chunks[0]["text"] == text


def test_long_opinion_produces_overlapping_chunks():
    """Opinion longer than CHUNK_SIZE → multiple overlapping chunks."""
    # Build a text of ~4000 chars (two chunks with overlap)
    sentences = [f"Sentence number {i} is here for padding purposes. " for i in range(80)]
    text = "".join(sentences)
    assert len(text) > 3000  # sanity check

    chunks = chunk_opinion(opinion_id=42, text=text, chunk_size=2000, overlap=400)
    assert len(chunks) >= 2

    # Chunks should overlap — end of chunk 0 overlaps with start of chunk 1
    end_of_first = chunks[0]["text"][-200:]
    assert end_of_first in chunks[1]["text"]

    # chunk_ids should be sequential
    for i, c in enumerate(chunks):
        assert c["chunk_index"] == i
        assert c["chunk_id"] == f"42_chunk_{i}"


def test_empty_text_returns_no_chunks():
    assert chunk_opinion(1, "", 2000, 400) == []
    assert chunk_opinion(1, "   ", 2000, 400) == []
    assert chunk_opinion(1, None, 2000, 400) == []


def test_chunks_never_split_mid_sentence():
    """Each chunk should end at a sentence boundary."""
    sentences = [f"Sentence {i} has some content here. " for i in range(100)]
    text = "".join(sentences)

    chunks = chunk_opinion(opinion_id=1, text=text, chunk_size=500, overlap=100)
    for c in chunks:
        # Each chunk should end with a period+space or period (end of text)
        stripped = c["text"].rstrip()
        assert stripped.endswith("."), f"Chunk does not end at sentence boundary: ...{stripped[-50:]}"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_chunk.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'chunk'`

**Step 3: Write `chunk.py`**

```python
# chunk.py
import logging
from typing import Optional

import nltk

logger = logging.getLogger(__name__)

# Ensure punkt tokenizer is available
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

from nltk.tokenize import sent_tokenize


def chunk_opinion(
    opinion_id: int,
    text: Optional[str],
    chunk_size: int = 2000,
    overlap: int = 400,
) -> list[dict]:
    """Split opinion text into overlapping chunks at sentence boundaries.

    Returns a list of dicts with keys: opinion_id, chunk_id, chunk_index, text.
    """
    if not text or not text.strip():
        return []

    text = text.strip()

    # If text fits in one chunk, return as-is
    if len(text) <= chunk_size:
        return [{
            "opinion_id": opinion_id,
            "chunk_id": f"{opinion_id}_chunk_0",
            "chunk_index": 0,
            "text": text,
        }]

    sentences = sent_tokenize(text)
    chunks = []
    current_sentences = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        # If adding this sentence exceeds chunk_size and we have content, finalize chunk
        if current_len + sentence_len > chunk_size and current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(chunk_text)

            # Build overlap: walk backward from end of current_sentences
            overlap_sentences = []
            overlap_len = 0
            for s in reversed(current_sentences):
                if overlap_len + len(s) > overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_len += len(s)

            current_sentences = overlap_sentences
            current_len = overlap_len

        current_sentences.append(sentence)
        current_len += sentence_len

    # Final chunk
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        chunks.append(chunk_text)

    return [
        {
            "opinion_id": opinion_id,
            "chunk_id": f"{opinion_id}_chunk_{i}",
            "chunk_index": i,
            "text": chunk_text,
        }
        for i, chunk_text in enumerate(chunks)
    ]
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_chunk.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add chunk.py tests/test_chunk.py
git commit -m "feat: opinion chunking with sentence-boundary splitting and overlap"
```

---

### Task 5: Embedding and Indexing (`embed.py`, `index.py`)

**Files:**
- Create: `embed.py`
- Create: `index.py`
- Create: `tests/test_embed.py`

**Step 1: Write the failing test**

```python
# tests/test_embed.py
import os
import pytest

os.environ["ML_LOCAL_DB"] = ":memory:"


def test_embed_chunks_returns_correct_shape():
    from embed import embed_chunks
    texts = ["This is sentence one.", "This is sentence two.", "Third sentence here."]
    embeddings = embed_chunks(texts)
    assert embeddings.shape == (3, 384)


def test_embed_chunks_normalized():
    """Embeddings should be L2-normalized (unit vectors)."""
    import numpy as np
    from embed import embed_chunks
    texts = ["Test sentence for normalization."]
    embeddings = embed_chunks(texts)
    norm = np.linalg.norm(embeddings[0])
    assert abs(norm - 1.0) < 0.01


def test_index_and_search_roundtrip(tmp_path):
    from index import build_collection, add_chunks_to_collection
    from embed import embed_chunks

    chunks = [
        {"chunk_id": "1_chunk_0", "opinion_id": 1, "chunk_index": 0,
         "text": "The debt collector called a third party and disclosed the debt.",
         "title": "Smith v. Collections", "court_name": "SD Texas",
         "court_type": "district", "circuit": "5th",
         "date_issued": "2024-01-15", "statutes": "FDCPA"},
        {"chunk_id": "2_chunk_0", "opinion_id": 2, "chunk_index": 0,
         "text": "The plaintiff received an autodialed robocall on their cell phone.",
         "title": "Jones v. Telecom", "court_name": "ND Ohio",
         "court_type": "district", "circuit": "6th",
         "date_issued": "2024-03-20", "statutes": "TCPA"},
    ]
    texts = [c["text"] for c in chunks]
    embeddings = embed_chunks(texts)

    collection = build_collection(str(tmp_path / "chroma_test"))
    add_chunks_to_collection(collection, chunks, embeddings)

    assert collection.count() == 2

    # Search for debt-related content — should rank chunk 1 first
    from search import search_opinions
    results = search_opinions(collection, "debt collector called third party", top_k=2)
    assert len(results) > 0
    assert results[0]["opinion_id"] == 1
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_embed.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'embed'`

**Step 3: Write `embed.py`**

```python
# embed.py
import logging
import os
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

import config

logger = logging.getLogger(__name__)

_model = None


def get_model():
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def embed_chunks(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Embed a list of text strings. Returns numpy array of shape (n, dim)."""
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 100,
        normalize_embeddings=True,
    )
    return embeddings


def save_checkpoint(embeddings: np.ndarray, chunk_ids: list[str], checkpoint_name: str):
    """Save embeddings to disk as a checkpoint."""
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    path = Path(config.CHECKPOINT_DIR) / f"{checkpoint_name}.npz"
    np.savez(path, embeddings=embeddings, chunk_ids=np.array(chunk_ids))
    logger.info(f"Saved checkpoint: {path}")


def load_checkpoint(checkpoint_name: str):
    """Load a saved checkpoint. Returns (embeddings, chunk_ids) or None."""
    path = Path(config.CHECKPOINT_DIR) / f"{checkpoint_name}.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return data["embeddings"], data["chunk_ids"].tolist()
```

**Step 4: Write `index.py`**

```python
# index.py
import logging

import chromadb
import numpy as np

import config

logger = logging.getLogger(__name__)

COLLECTION_NAME = "federal_opinions"


def build_collection(chroma_dir: str = None):
    """Create or open the ChromaDB collection."""
    if chroma_dir is None:
        chroma_dir = config.CHROMA_DIR
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": config.EMBEDDING_MODEL,
        },
    )
    # Check for model mismatch
    stored_model = collection.metadata.get("embedding_model")
    if stored_model and stored_model != config.EMBEDDING_MODEL:
        raise ValueError(
            f"Model mismatch: collection was built with '{stored_model}' "
            f"but config specifies '{config.EMBEDDING_MODEL}'. "
            f"Delete {chroma_dir} and re-index to switch models."
        )
    return collection


def add_chunks_to_collection(
    collection,
    chunks: list[dict],
    embeddings: np.ndarray,
    batch_size: int = 5000,
):
    """Add chunks and their embeddings to ChromaDB."""
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_embeddings = embeddings[i:i + batch_size]

        collection.add(
            ids=[c["chunk_id"] for c in batch],
            embeddings=batch_embeddings.tolist(),
            documents=[c["text"] for c in batch],
            metadatas=[
                {
                    "opinion_id": c["opinion_id"],
                    "title": c.get("title", ""),
                    "court_name": c.get("court_name", ""),
                    "court_type": c.get("court_type", ""),
                    "circuit": c.get("circuit", ""),
                    "date_issued": c.get("date_issued", ""),
                    "statutes": c.get("statutes", ""),
                    "chunk_index": c["chunk_index"],
                }
                for c in batch
            ],
        )
        logger.info(f"Indexed batch {i // batch_size + 1} ({len(batch)} chunks)")
```

**Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_embed.py -v`
Expected: All 3 tests PASS (the roundtrip test depends on `search.py` from Task 6 — move it to Task 6 if needed, or stub `search_opinions` here)

Note: The roundtrip test references `search.search_opinions` which doesn't exist yet. **Move `test_index_and_search_roundtrip` to `tests/test_search.py` in Task 6.** Run only the first two tests here:

Run: `python -m pytest tests/test_embed.py::test_embed_chunks_returns_correct_shape tests/test_embed.py::test_embed_chunks_normalized -v`
Expected: PASS

**Step 6: Commit**

```bash
git add embed.py index.py tests/test_embed.py
git commit -m "feat: embedding and ChromaDB indexing with checkpoints and model mismatch detection"
```

---

### Task 6: Search Engine (`search.py`)

**Files:**
- Create: `search.py`
- Create: `tests/test_search.py`

**Step 1: Write the failing test**

```python
# tests/test_search.py
import os
import pytest

os.environ["ML_LOCAL_DB"] = ":memory:"


@pytest.fixture
def populated_collection(tmp_path):
    """Build a ChromaDB collection with test data."""
    from index import build_collection, add_chunks_to_collection
    from embed import embed_chunks

    chunks = [
        {"chunk_id": "1_chunk_0", "opinion_id": 1, "chunk_index": 0,
         "text": "The debt collector violated the FDCPA by calling a third party and disclosing the consumer's debt to the neighbor.",
         "title": "Smith v. Collections Inc", "court_name": "SD Texas",
         "court_type": "district", "circuit": "5th",
         "date_issued": "2024-01-15", "statutes": "FDCPA"},
        {"chunk_id": "1_chunk_1", "opinion_id": 1, "chunk_index": 1,
         "text": "The court awarded statutory damages of one thousand dollars under section 1692k.",
         "title": "Smith v. Collections Inc", "court_name": "SD Texas",
         "court_type": "district", "circuit": "5th",
         "date_issued": "2024-01-15", "statutes": "FDCPA"},
        {"chunk_id": "2_chunk_0", "opinion_id": 2, "chunk_index": 0,
         "text": "Plaintiff received multiple autodialed robocalls on their cellular telephone in violation of the TCPA.",
         "title": "Jones v. Telecom Corp", "court_name": "ND Ohio",
         "court_type": "district", "circuit": "6th",
         "date_issued": "2024-03-20", "statutes": "TCPA"},
    ]
    texts = [c["text"] for c in chunks]
    embeddings = embed_chunks(texts)

    collection = build_collection(str(tmp_path / "chroma_test"))
    add_chunks_to_collection(collection, chunks, embeddings)
    return collection


def test_search_returns_results(populated_collection):
    from search import search_opinions
    results = search_opinions(populated_collection, "debt collector called third party")
    assert len(results) > 0


def test_search_groups_by_opinion(populated_collection):
    """Two chunks from opinion 1 should be grouped into one result."""
    from search import search_opinions
    results = search_opinions(populated_collection, "debt collector damages", top_k=10)
    opinion_ids = [r["opinion_id"] for r in results]
    # No duplicate opinion_ids
    assert len(opinion_ids) == len(set(opinion_ids))


def test_search_best_passage_selected(populated_collection):
    from search import search_opinions
    results = search_opinions(populated_collection, "robocall cellular telephone TCPA")
    # TCPA-related query should rank the TCPA opinion first
    assert results[0]["opinion_id"] == 2


def test_search_with_statute_filter(populated_collection):
    from search import search_opinions
    results = search_opinions(
        populated_collection, "violation", filters={"statute": "TCPA"}
    )
    for r in results:
        assert "TCPA" in r["statutes"]
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_search.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'search'`

**Step 3: Write `search.py`**

```python
# search.py
import argparse
import logging
from collections import defaultdict

from embed import embed_chunks
from index import build_collection
import config

logger = logging.getLogger(__name__)


def search_opinions(
    collection,
    query: str,
    top_k: int = None,
    filters: dict = None,
) -> list[dict]:
    """Search for opinions matching the query.

    Returns a list of dicts, one per opinion, sorted by best match.
    Each dict: opinion_id, title, court_name, circuit, date_issued,
               statutes, similarity_score, best_passage, chunk_index.
    """
    if top_k is None:
        top_k = config.DEFAULT_TOP_K

    n_results = top_k * config.OVERSAMPLE_FACTOR

    # Build ChromaDB where clause from filters
    where = _build_where(filters)

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where if where else None,
        include=["documents", "metadatas", "distances"],
    )

    # Group by opinion_id, keep best chunk per opinion
    return _group_results(results, top_k)


def _build_where(filters: dict = None) -> dict:
    """Convert filter dict to ChromaDB where clause."""
    if not filters:
        return None

    conditions = []

    if filters.get("statute"):
        conditions.append({"statutes": {"$contains": filters["statute"]}})

    if filters.get("circuit"):
        conditions.append({"circuit": filters["circuit"]})

    if filters.get("court_type"):
        conditions.append({"court_type": filters["court_type"]})

    if filters.get("date_from"):
        conditions.append({"date_issued": {"$gte": filters["date_from"]}})

    if filters.get("date_to"):
        conditions.append({"date_issued": {"$lte": filters["date_to"]}})

    if filters.get("opinion_ids"):
        conditions.append({"opinion_id": {"$in": filters["opinion_ids"]}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _group_results(results: dict, top_k: int) -> list[dict]:
    """Group ChromaDB results by opinion, keeping the best chunk per opinion."""
    if not results["ids"][0]:
        return []

    # Collect all chunks grouped by opinion_id
    opinion_best = {}

    for i, chunk_id in enumerate(results["ids"][0]):
        meta = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        document = results["documents"][0][i]
        opinion_id = meta["opinion_id"]

        # Cosine distance → similarity: similarity = 1 - distance
        similarity = 1.0 - distance

        if opinion_id not in opinion_best or similarity > opinion_best[opinion_id]["similarity_score"]:
            opinion_best[opinion_id] = {
                "opinion_id": opinion_id,
                "title": meta.get("title", ""),
                "court_name": meta.get("court_name", ""),
                "court_type": meta.get("court_type", ""),
                "circuit": meta.get("circuit", ""),
                "date_issued": meta.get("date_issued", ""),
                "statutes": meta.get("statutes", ""),
                "similarity_score": similarity,
                "best_passage": document,
                "chunk_index": meta.get("chunk_index", 0),
            }

    # Sort by similarity descending, take top_k
    ranked = sorted(opinion_best.values(), key=lambda x: x["similarity_score"], reverse=True)
    return ranked[:top_k]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search federal court opinions")
    parser.add_argument("query", help="Natural language search query")
    parser.add_argument("--statute", help="Filter by statute (FDCPA, TCPA, FCRA)")
    parser.add_argument("--circuit", help="Filter by circuit (e.g., 5th)")
    parser.add_argument("--top", type=int, default=10, help="Number of results")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    collection = build_collection()

    filters = {}
    if args.statute:
        filters["statute"] = args.statute
    if args.circuit:
        filters["circuit"] = args.circuit

    results = search_opinions(collection, args.query, top_k=args.top, filters=filters)

    for i, r in enumerate(results, 1):
        print(f"\n--- Result {i} (score: {r['similarity_score']:.3f}) ---")
        print(f"  {r['title']}")
        print(f"  {r['court_name']} | {r['circuit']} Circuit | {r['date_issued']}")
        print(f"  Statutes: {r['statutes']}")
        print(f"  Passage: {r['best_passage'][:200]}...")
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_search.py -v`
Expected: All 4 tests PASS

**Step 5: Also move the roundtrip test from test_embed.py**

Remove `test_index_and_search_roundtrip` from `tests/test_embed.py` if it was added there. The roundtrip is now covered by `tests/test_search.py`.

**Step 6: Commit**

```bash
git add search.py tests/test_search.py
git commit -m "feat: search engine with filtering, grouping, and CLI"
```

---

### Task 7: Processing Pipeline (tie it all together)

**Files:**
- Create: `pipeline.py`

This is the script that runs the full pipeline: sync → chunk → embed → index. No separate test — we validate by running it against the real (or test) data and checking ChromaDB counts.

**Step 1: Write `pipeline.py`**

```python
# pipeline.py
"""
Full processing pipeline: sync → chunk → embed → index.

Usage:
    python pipeline.py              # process all new opinions
    python pipeline.py --sync-only  # just sync, don't embed
    python pipeline.py --reindex    # re-chunk and re-embed everything
"""
import argparse
import logging
import os
from pathlib import Path

import numpy as np
from sqlalchemy import text

import config
from db import init_local_db, get_local_engine, get_session, Opinion, OpinionStatute, Statute
from sync import sync_from_source
from chunk import chunk_opinion
from embed import embed_chunks, save_checkpoint
from index import build_collection, add_chunks_to_collection

logger = logging.getLogger(__name__)

CHECKPOINT_BATCH = 500  # opinions per checkpoint


def run_pipeline(sync_only=False, reindex=False):
    # Ensure data dirs exist
    os.makedirs(config.CHROMA_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # Step 1: Sync
    engine = get_local_engine()
    init_local_db(engine)
    logger.info("Syncing from source database...")
    new_count = sync_from_source(local_engine=engine)
    logger.info(f"Sync complete: {new_count} new opinions")

    if sync_only:
        return

    # Step 2: Find opinions to process
    session = get_session(engine)
    if reindex:
        # Reset all chunked flags
        session.execute(text("UPDATE opinions SET chunked = 0"))
        session.commit()

    opinions = session.execute(text(
        "SELECT id, plain_text FROM opinions WHERE chunked = 0 AND plain_text IS NOT NULL AND plain_text != ''"
    )).fetchall()
    logger.info(f"Opinions to process: {len(opinions)}")

    if not opinions:
        logger.info("Nothing to process")
        session.close()
        return

    # Step 3: Chunk + Embed + Index in batches
    collection = build_collection()

    # Build statute lookup for metadata
    statute_map = {}
    rows = session.execute(text(
        "SELECT os.opinion_id, s.key FROM opinion_statutes os JOIN statutes s ON os.statute_id = s.id"
    )).fetchall()
    for opinion_id, key in rows:
        statute_map.setdefault(opinion_id, []).append(key.upper())

    # Build opinion metadata lookup
    meta_rows = session.execute(text(
        "SELECT id, title, court_name, court_type, circuit, date_issued FROM opinions WHERE chunked = 0"
    )).fetchall()
    meta_map = {
        r[0]: {"title": r[1], "court_name": r[2], "court_type": r[3],
               "circuit": r[4] or "", "date_issued": r[5] or ""}
        for r in meta_rows
    }

    total_chunks = 0
    processed = 0

    for batch_start in range(0, len(opinions), CHECKPOINT_BATCH):
        batch = opinions[batch_start:batch_start + CHECKPOINT_BATCH]
        all_chunks = []

        for opinion_id, plain_text in batch:
            chunks = chunk_opinion(
                opinion_id=opinion_id,
                text=plain_text,
                chunk_size=config.CHUNK_SIZE,
                overlap=config.CHUNK_OVERLAP,
            )

            # Attach metadata to each chunk
            meta = meta_map.get(opinion_id, {})
            statutes_str = ",".join(sorted(statute_map.get(opinion_id, [])))
            for c in chunks:
                c.update({
                    "title": meta.get("title", ""),
                    "court_name": meta.get("court_name", ""),
                    "court_type": meta.get("court_type", ""),
                    "circuit": meta.get("circuit", ""),
                    "date_issued": meta.get("date_issued", ""),
                    "statutes": statutes_str,
                })
            all_chunks.extend(chunks)

        if not all_chunks:
            continue

        # Embed
        texts = [c["text"] for c in all_chunks]
        logger.info(f"Embedding {len(texts)} chunks (batch {batch_start // CHECKPOINT_BATCH + 1})...")
        embeddings = embed_chunks(texts)

        # Save checkpoint
        checkpoint_name = f"batch_{batch_start}"
        save_checkpoint(embeddings, [c["chunk_id"] for c in all_chunks], checkpoint_name)

        # Index
        add_chunks_to_collection(collection, all_chunks, embeddings)

        # Mark opinions as chunked
        opinion_ids = [oid for oid, _ in batch]
        placeholders = ",".join(str(oid) for oid in opinion_ids)
        session.execute(text(f"UPDATE opinions SET chunked = 1 WHERE id IN ({placeholders})"))
        session.commit()

        total_chunks += len(all_chunks)
        processed += len(batch)
        logger.info(f"Progress: {processed}/{len(opinions)} opinions, {total_chunks} total chunks")

    session.close()
    logger.info(f"Pipeline complete. {processed} opinions → {total_chunks} chunks indexed.")
    logger.info(f"ChromaDB collection count: {collection.count()}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Path(config.PROJECT_ROOT) / "data" / "pipeline.log"),
        ],
    )

    parser = argparse.ArgumentParser(description="Run the full processing pipeline")
    parser.add_argument("--sync-only", action="store_true", help="Only sync, don't embed")
    parser.add_argument("--reindex", action="store_true", help="Re-process all opinions")
    args = parser.parse_args()

    run_pipeline(sync_only=args.sync_only, reindex=args.reindex)
```

**Step 2: Test by running against real data (small sample)**

Run:
```bash
python pipeline.py --sync-only
```
Expected: Should sync ~15K opinions from cld, print count.

Then test a small search:
```bash
python search.py "debt collector harassment"
```
Expected: Should fail gracefully (no embeddings yet) or return empty.

**Step 3: Commit**

```bash
git add pipeline.py
git commit -m "feat: full processing pipeline - sync, chunk, embed, index"
```

---

### Task 8: Streamlit UI — App Shell and Placeholder Pages

**Files:**
- Create: `app.py`
- Create: `pages/1_Search.py`
- Create: `pages/2_Topics.py`
- Create: `pages/3_Analytics.py`
- Create: `pages/4_Citations.py`

**Step 1: Create `app.py`**

```python
# app.py
import streamlit as st

st.set_page_config(
    page_title="Federal Opinion Search",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Federal Opinion Intelligence Platform")
st.markdown(
    "Search and analyze ~30,000 federal court opinions across FDCPA, TCPA, and FCRA."
)
st.markdown("Use the sidebar to navigate between tools.")
```

**Step 2: Create placeholder pages**

```python
# pages/2_Topics.py
import streamlit as st

st.set_page_config(page_title="Topics", page_icon="⚖️", layout="wide")
st.title("Topic Modeling")
st.info("Coming soon — discover natural themes and clusters across the corpus using BERTopic.")
```

```python
# pages/3_Analytics.py
import streamlit as st

st.set_page_config(page_title="Analytics", page_icon="⚖️", layout="wide")
st.title("Trends & Analytics")
st.info("Coming soon — track outcomes, damages, and legal reasoning trends over time, by court, and by judge.")
```

```python
# pages/4_Citations.py
import streamlit as st

st.set_page_config(page_title="Citations", page_icon="⚖️", layout="wide")
st.title("Citation Network")
st.info("Coming soon — map how opinions cite each other to find influential cases and emerging trends.")
```

**Step 3: Commit**

```bash
git add app.py pages/
git commit -m "feat: Streamlit app shell with placeholder pages"
```

---

### Task 9: Streamlit Search Page

**Files:**
- Create: `pages/1_Search.py`

This is the main search interface. It's the largest single file in the project.

**Step 1: Write `pages/1_Search.py`**

```python
# pages/1_Search.py
import streamlit as st
from sqlalchemy import text

from db import get_local_engine, get_session, init_local_db
from index import build_collection
from search import search_opinions


@st.cache_resource
def get_collection():
    return build_collection()


@st.cache_resource
def get_db_engine():
    engine = get_local_engine()
    init_local_db(engine)
    return engine


def get_filter_options(engine):
    """Load filter options from the local DB."""
    session = get_session(engine)
    circuits = sorted([
        r[0] for r in session.execute(
            text("SELECT DISTINCT circuit FROM opinions WHERE circuit IS NOT NULL AND circuit != ''")
        ).fetchall()
    ])
    courts = sorted(set(
        r[0] for r in session.execute(
            text("SELECT DISTINCT court_name FROM opinions WHERE court_name IS NOT NULL")
        ).fetchall()
    ))
    subsections = sorted([
        r[0] for r in session.execute(
            text("SELECT DISTINCT subsection FROM fdcpa_sections ORDER BY subsection")
        ).fetchall()
    ])
    session.close()
    return circuits, courts, subsections


def get_full_opinion_text(engine, opinion_id: int) -> str:
    """Retrieve the full opinion text from the local DB."""
    session = get_session(engine)
    row = session.execute(
        text("SELECT plain_text FROM opinions WHERE id = :oid"),
        {"oid": opinion_id},
    ).fetchone()
    session.close()
    return row[0] if row else ""


def get_opinion_ids_for_subsections(engine, subsections: list[str]) -> list[int]:
    """Get opinion IDs that cite any of the given FDCPA subsections."""
    session = get_session(engine)
    placeholders = ",".join(f"'{s}'" for s in subsections)
    rows = session.execute(text(
        f"SELECT DISTINCT opinion_id FROM fdcpa_sections WHERE subsection IN ({placeholders})"
    )).fetchall()
    session.close()
    return [r[0] for r in rows]


# --- Page Config ---
st.set_page_config(page_title="Search", page_icon="⚖️", layout="wide")
st.title("Semantic Search")

engine = get_db_engine()
collection = get_collection()

# --- Sidebar Filters ---
with st.sidebar:
    st.header("Filters")

    statute_options = ["FDCPA", "TCPA", "FCRA"]
    selected_statutes = st.multiselect("Statute", statute_options)

    circuits, courts, subsections = get_filter_options(engine)
    selected_circuits = st.multiselect("Circuit", circuits)

    court_type = st.radio("Court Type", ["All", "District", "Circuit"], horizontal=True)

    # Date range
    col1, col2 = st.columns(2)
    with col1:
        date_from = st.text_input("Date from", placeholder="YYYY-MM-DD")
    with col2:
        date_to = st.text_input("Date to", placeholder="YYYY-MM-DD")

    # FDCPA subsection filter (only if FDCPA selected or no statute filter)
    selected_subsections = []
    if subsections and ("FDCPA" in selected_statutes or not selected_statutes):
        selected_subsections = st.multiselect("FDCPA Subsections", subsections)

    results_per_page = st.slider("Results per page", 5, 50, 10)

# --- Search Box ---
query = st.text_input(
    "Search opinions",
    placeholder="e.g., collector called third party and disclosed debt",
    label_visibility="collapsed",
)

# --- Execute Search ---
if query:
    # Build filters
    filters = {}

    if selected_statutes and len(selected_statutes) == 1:
        filters["statute"] = selected_statutes[0]

    if selected_circuits and len(selected_circuits) == 1:
        filters["circuit"] = selected_circuits[0]

    if court_type != "All":
        filters["court_type"] = court_type.lower()

    if date_from:
        filters["date_from"] = date_from
    if date_to:
        filters["date_to"] = date_to

    # Handle FDCPA subsection filter via SQLite lookup
    if selected_subsections:
        opinion_ids = get_opinion_ids_for_subsections(engine, selected_subsections)
        if opinion_ids:
            filters["opinion_ids"] = opinion_ids
        else:
            st.warning("No opinions match the selected subsections.")
            st.stop()

    # Handle multi-statute and multi-circuit filtering client-side
    multi_statute_filter = selected_statutes if len(selected_statutes) > 1 else None
    multi_circuit_filter = selected_circuits if len(selected_circuits) > 1 else None

    # Oversample if we need to filter client-side
    effective_top_k = results_per_page
    if multi_statute_filter or multi_circuit_filter:
        effective_top_k = results_per_page * 3

    with st.spinner("Searching..."):
        results = search_opinions(collection, query, top_k=effective_top_k, filters=filters)

    # Client-side filtering for multi-select
    if multi_statute_filter:
        results = [r for r in results if any(s in r["statutes"] for s in multi_statute_filter)]
    if multi_circuit_filter:
        results = [r for r in results if r["circuit"] in multi_circuit_filter]

    results = results[:results_per_page]

    # --- Display Results ---
    if not results:
        st.info("No results found. Try broadening your search or adjusting filters.")
    else:
        st.markdown(f"**{len(results)} results**")

        for i, r in enumerate(results):
            score_pct = r["similarity_score"] * 100

            # Result header
            st.markdown(f"### {r['title']}")

            # Metadata row
            meta_parts = []
            if r["court_name"]:
                meta_parts.append(r["court_name"])
            if r["circuit"]:
                meta_parts.append(f"{r['circuit']} Circuit")
            if r["date_issued"]:
                meta_parts.append(r["date_issued"])
            st.caption(" | ".join(meta_parts))

            # Statute badges
            if r["statutes"]:
                statute_list = r["statutes"].split(",")
                st.markdown(" ".join(f"`{s}`" for s in statute_list))

            # Similarity score bar
            st.progress(score_pct / 100, text=f"Relevance: {score_pct:.1f}%")

            # Best matching passage
            st.markdown(f"> {r['best_passage'][:500]}{'...' if len(r['best_passage']) > 500 else ''}")

            # Expandable full text
            with st.expander("Show full opinion"):
                full_text = get_full_opinion_text(engine, r["opinion_id"])
                if full_text:
                    st.text(full_text)
                else:
                    st.warning("Full text not available.")

            st.divider()

elif collection.count() == 0:
    st.warning("No opinions indexed yet. Run `python pipeline.py` to process opinions.")
else:
    st.info(f"Ready to search {collection.count():,} chunks from the federal opinion corpus.")
```

**Step 2: Test the UI**

Run: `streamlit run app.py`
Expected: App loads with sidebar nav showing Search, Topics, Analytics, Citations. Search page shows filters and search box. Placeholder pages show "Coming soon" messages.

**Step 3: Commit**

```bash
git add pages/1_Search.py
git commit -m "feat: Streamlit search page with filters, results, and full-text expand"
```

---

### Task 10: End-to-End Validation

This is not a code task — it's a verification step.

**Step 1: Run the full pipeline on available data**

```bash
python pipeline.py
```

Expected: Syncs ~15K opinions, chunks them, embeds (~300K chunks, takes 2-3 hours on CPU), indexes in ChromaDB. Watch the logs for errors.

If 2-3 hours is too long for initial testing, do a small test first:

```bash
python -c "
from db import get_local_engine, init_local_db, get_session
from sqlalchemy import text

engine = get_local_engine()
init_local_db(engine)
session = get_session(engine)

# Temporarily limit to 100 opinions for testing
session.execute(text('UPDATE opinions SET chunked = 1'))
session.execute(text('UPDATE opinions SET chunked = 0 WHERE id IN (SELECT id FROM opinions WHERE plain_text IS NOT NULL LIMIT 100)'))
session.commit()
session.close()
"
python pipeline.py
```

**Step 2: Test search via CLI**

```bash
python search.py "debt collector called third party and disclosed debt" --statute FDCPA --top 5
python search.py "autodialer robocall cell phone" --statute TCPA --top 5
python search.py "credit report inaccurate information" --statute FCRA --top 5
```

Expected: Relevant results for each query, properly grouped by opinion.

**Step 3: Test the UI**

```bash
streamlit run app.py
```

Test:
- Search with no filters
- Search with statute filter
- Search with circuit filter
- Search with date range
- Expand a result to see full opinion text
- Check placeholder tabs load

**Step 4: Commit any fixes**

```bash
git add -A
git commit -m "fix: end-to-end validation fixes"
```

---

## Summary

| Task | Module | What It Builds |
|------|--------|----------------|
| 1 | `config.py`, `requirements.txt` | Project setup |
| 2 | `db.py` | Local SQLite schema |
| 3 | `sync.py` | Pull opinions from cld |
| 4 | `chunk.py` | Text chunking |
| 5 | `embed.py`, `index.py` | Embedding + ChromaDB |
| 6 | `search.py` | Search engine + CLI |
| 7 | `pipeline.py` | Orchestration script |
| 8 | `app.py`, placeholder pages | Streamlit shell |
| 9 | `pages/1_Search.py` | Search UI |
| 10 | — | End-to-end validation |
