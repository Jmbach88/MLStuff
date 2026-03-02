# Citation Network Analysis — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extract case citations from 30K federal court opinions, resolve them to corpus IDs, build a directed citation graph, compute network metrics, and visualize in a Citations dashboard.

**Architecture:** Regex-based extraction from `plain_text`, two-pass resolution (volume/reporter/page lookup + WL/LEXIS matching), NetworkX directed graph for metrics (PageRank, HITS, Louvain communities), results stored in new Citation and OpinionMetric tables.

**Tech Stack:** networkx, regex (stdlib), SQLAlchemy ORM, Streamlit + Plotly

---

### Task 1: Add networkx dependency and DB schema

**Files:**
- Modify: `requirements.txt`
- Modify: `db.py` (after line 91, before `get_local_engine`)

**Step 1: Add networkx to requirements.txt**

Append to `requirements.txt`:
```
networkx>=3.0
```

**Step 2: Run `pip install networkx>=3.0`**

Run: `pip install networkx>=3.0`
Expected: Successfully installed

**Step 3: Add Citation and OpinionMetric ORM classes to db.py**

Add after the `Model` class (line 91) in `db.py`:

```python
class Citation(Base):
    __tablename__ = "citations"
    id = Column(Integer, primary_key=True, autoincrement=True)
    citing_opinion_id = Column(Integer, ForeignKey("opinions.id"), nullable=False)
    cited_opinion_id = Column(Integer, ForeignKey("opinions.id"), nullable=True)
    volume = Column(Text)
    reporter = Column(Text)
    page = Column(Text)
    citation_string = Column(Text, nullable=False)
    context_snippet = Column(Text)
    __table_args__ = (
        UniqueConstraint("citing_opinion_id", "volume", "reporter", "page",
                         name="uq_citation_dedup"),
    )


class OpinionMetric(Base):
    __tablename__ = "opinion_metrics"
    opinion_id = Column(Integer, ForeignKey("opinions.id"), primary_key=True)
    in_degree = Column(Integer, default=0)
    out_degree = Column(Integer, default=0)
    pagerank = Column(Float)
    hub_score = Column(Float)
    authority_score = Column(Float)
    community_id = Column(Integer)
```

**Step 4: Verify tables create correctly**

Run: `python -c "from db import get_local_engine, init_local_db; init_local_db(get_local_engine()); print('OK')"`
Expected: `OK`

**Step 5: Commit**

```bash
git add requirements.txt db.py
git commit -m "feat: add Citation and OpinionMetric tables, networkx dependency"
```

---

### Task 2: Citation extraction — tests first

**Files:**
- Create: `tests/test_citations.py`
- Create: `citations.py` (stub)

**Step 1: Write failing tests for citation extraction**

Create `tests/test_citations.py`:

```python
import os
import pytest

os.environ["ML_LOCAL_DB"] = ":memory:"

from db import get_local_engine, init_local_db, get_session, Opinion, Citation


class TestCitationExtraction:
    """Tests for extract_citations_from_text()."""

    def test_extracts_f2d_citation(self):
        from citations import extract_citations_from_text
        text = "The court relied on Smith v. Jones, 123 F.2d 456 (2d Cir. 1990)."
        cites = extract_citations_from_text(text)
        assert len(cites) == 1
        assert cites[0]["volume"] == "123"
        assert cites[0]["reporter"] == "F.2d"
        assert cites[0]["page"] == "456"

    def test_extracts_f3d_citation(self):
        from citations import extract_citations_from_text
        text = "See also Doe v. Roe, 789 F.3d 101 (9th Cir. 2015)."
        cites = extract_citations_from_text(text)
        assert len(cites) == 1
        assert cites[0]["volume"] == "789"
        assert cites[0]["reporter"] == "F.3d"
        assert cites[0]["page"] == "101"

    def test_extracts_f4th_citation(self):
        from citations import extract_citations_from_text
        text = "In Brown v. Green, 45 F.4th 678 (5th Cir. 2022), the court held..."
        cites = extract_citations_from_text(text)
        assert len(cites) == 1
        assert cites[0]["volume"] == "45"
        assert cites[0]["reporter"] == "F.4th"
        assert cites[0]["page"] == "678"

    def test_extracts_f_first_series(self):
        from citations import extract_citations_from_text
        text = "The seminal case is 100 F. 200."
        cites = extract_citations_from_text(text)
        assert len(cites) == 1
        assert cites[0]["reporter"] == "F."

    def test_extracts_f_supp_citations(self):
        from citations import extract_citations_from_text
        text = "See 100 F. Supp. 200 and 300 F. Supp. 2d 400 and 500 F. Supp. 3d 600."
        cites = extract_citations_from_text(text)
        assert len(cites) == 3
        reporters = {c["reporter"] for c in cites}
        assert reporters == {"F. Supp.", "F. Supp. 2d", "F. Supp. 3d"}

    def test_extracts_us_reports(self):
        from citations import extract_citations_from_text
        text = "The Supreme Court in 410 U.S. 113 held that..."
        cites = extract_citations_from_text(text)
        assert len(cites) == 1
        assert cites[0]["reporter"] == "U.S."
        assert cites[0]["volume"] == "410"
        assert cites[0]["page"] == "113"

    def test_extracts_s_ct(self):
        from citations import extract_citations_from_text
        text = "See 123 S. Ct. 456 and 789 S.Ct. 101."
        cites = extract_citations_from_text(text)
        assert len(cites) == 2

    def test_extracts_westlaw_cite(self):
        from citations import extract_citations_from_text
        text = "See Smith v. Jones, 2024 WL 1234567 (S.D.N.Y. 2024)."
        cites = extract_citations_from_text(text)
        assert len(cites) == 1
        assert cites[0]["volume"] == "2024"
        assert cites[0]["reporter"] == "WL"
        assert cites[0]["page"] == "1234567"

    def test_extracts_lexis_cites(self):
        from citations import extract_citations_from_text
        text = "See 2023 U.S. Dist. LEXIS 12345 and 2022 U.S. App. LEXIS 67890."
        cites = extract_citations_from_text(text)
        assert len(cites) == 2
        reporters = {c["reporter"] for c in cites}
        assert "U.S. Dist. LEXIS" in reporters
        assert "U.S. App. LEXIS" in reporters

    def test_captures_context_snippet(self):
        from citations import extract_citations_from_text
        text = "X" * 200 + "cited 123 F.2d 456 here" + "Y" * 200
        cites = extract_citations_from_text(text)
        assert len(cites) == 1
        # Context should be ~100 chars on each side of the citation
        assert len(cites[0]["context_snippet"]) <= 250
        assert "123 F.2d 456" in cites[0]["context_snippet"]

    def test_deduplicates_same_citation(self):
        from citations import extract_citations_from_text
        text = "See 123 F.2d 456. The court in 123 F.2d 456 also held..."
        cites = extract_citations_from_text(text)
        assert len(cites) == 1

    def test_multiple_different_citations(self):
        from citations import extract_citations_from_text
        text = "See 123 F.2d 456 and 789 F.3d 101."
        cites = extract_citations_from_text(text)
        assert len(cites) == 2

    def test_empty_text_returns_empty(self):
        from citations import extract_citations_from_text
        assert extract_citations_from_text("") == []
        assert extract_citations_from_text(None) == []


class TestCitationDedup:
    """Tests for dedup UniqueConstraint at the DB level."""

    def test_db_dedup_constraint(self):
        from citations import store_citations
        engine = get_local_engine()
        init_local_db(engine)
        session = get_session(engine)

        # Create a test opinion
        from sqlalchemy import text as sql_text
        session.execute(sql_text("DELETE FROM citations"))
        session.execute(sql_text("DELETE FROM opinions"))
        session.commit()
        session.add(Opinion(id=1, package_id="pkg_1", title="Case 1"))
        session.commit()
        session.close()

        citations = [
            {"volume": "123", "reporter": "F.2d", "page": "456",
             "citation_string": "123 F.2d 456", "context_snippet": "context"},
        ]
        store_citations(engine, 1, citations)

        session = get_session(engine)
        count = session.query(Citation).filter_by(citing_opinion_id=1).count()
        session.close()
        assert count == 1
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_citations.py -v`
Expected: FAIL (citations module not found)

**Step 3: Commit failing tests**

```bash
git add tests/test_citations.py
git commit -m "test: add citation extraction tests (red)"
```

---

### Task 3: Citation extraction — implementation

**Files:**
- Create: `citations.py`

**Step 1: Implement citations.py with extraction and storage**

Create `citations.py`:

```python
"""
Citation network analysis: extraction, resolution, graph metrics.

Usage:
    python citations.py                # extract + resolve + metrics
    python citations.py --extract-only # just extract to DB
    python citations.py --metrics-only # recompute from existing citations
    python citations.py --info         # print summary
"""
import argparse
import logging
import re
from collections import defaultdict

import networkx as nx
from sqlalchemy import text

import config
from db import (
    get_local_engine, init_local_db, get_session,
    Opinion, Citation, OpinionMetric,
)

logger = logging.getLogger(__name__)

# Regex patterns for federal case citations.
# Order matters: longer/more-specific reporters must come before shorter ones
# to prevent partial matches (e.g., "F. Supp. 3d" before "F. Supp." before "F.").
CITATION_PATTERNS = [
    # F. Supp. series (most specific first)
    (r'(\d+)\s+F\.\s*Supp\.\s*3d\s+(\d+)', 'F. Supp. 3d'),
    (r'(\d+)\s+F\.\s*Supp\.\s*2d\s+(\d+)', 'F. Supp. 2d'),
    (r'(\d+)\s+F\.\s*Supp\.\s+(\d+)', 'F. Supp.'),
    # Federal Reporter series
    (r'(\d+)\s+F\.4th\s+(\d+)', 'F.4th'),
    (r'(\d+)\s+F\.3d\s+(\d+)', 'F.3d'),
    (r'(\d+)\s+F\.2d\s+(\d+)', 'F.2d'),
    (r'(\d+)\s+F\.\s+(\d+)', 'F.'),
    # Supreme Court
    (r'(\d+)\s+U\.S\.\s+(\d+)', 'U.S.'),
    (r'(\d+)\s+S\.\s?Ct\.\s+(\d+)', 'S. Ct.'),
    # Westlaw
    (r'(\d{4})\s+WL\s+(\d+)', 'WL'),
    # LexisNexis
    (r'(\d{4})\s+U\.S\.\s+Dist\.\s+LEXIS\s+(\d+)', 'U.S. Dist. LEXIS'),
    (r'(\d{4})\s+U\.S\.\s+App\.\s+LEXIS\s+(\d+)', 'U.S. App. LEXIS'),
]

# Pre-compile all patterns
_COMPILED_PATTERNS = [(re.compile(p), r) for p, r in CITATION_PATTERNS]

CONTEXT_CHARS = 100  # chars of surrounding context to capture


def extract_citations_from_text(text_content):
    """Extract federal case citations from plain text.

    Returns list of dicts with keys:
        volume, reporter, page, citation_string, context_snippet
    Deduplicates by (volume, reporter, page).
    """
    if not text_content:
        return []

    seen = set()
    results = []

    for pattern, reporter_name in _COMPILED_PATTERNS:
        for match in pattern.finditer(text_content):
            volume = match.group(1)
            page = match.group(2)
            key = (volume, reporter_name, page)

            if key in seen:
                continue
            seen.add(key)

            # Extract context snippet
            start = max(0, match.start() - CONTEXT_CHARS)
            end = min(len(text_content), match.end() + CONTEXT_CHARS)
            context = text_content[start:end].strip()

            results.append({
                "volume": volume,
                "reporter": reporter_name,
                "page": page,
                "citation_string": match.group(0).strip(),
                "context_snippet": context,
            })

    return results


def store_citations(engine, opinion_id, citations):
    """Store extracted citations for one opinion. Skips duplicates."""
    if not citations:
        return 0
    session = get_session(engine)
    stored = 0
    try:
        for cite in citations:
            existing = session.query(Citation).filter_by(
                citing_opinion_id=opinion_id,
                volume=cite["volume"],
                reporter=cite["reporter"],
                page=cite["page"],
            ).first()
            if existing:
                continue
            session.add(Citation(
                citing_opinion_id=opinion_id,
                volume=cite["volume"],
                reporter=cite["reporter"],
                page=cite["page"],
                citation_string=cite["citation_string"],
                context_snippet=cite["context_snippet"],
            ))
            stored += 1
        session.commit()
    finally:
        session.close()
    return stored
```

**Step 2: Run extraction tests**

Run: `python -m pytest tests/test_citations.py::TestCitationExtraction -v`
Expected: All PASS

Run: `python -m pytest tests/test_citations.py::TestCitationDedup -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add citations.py
git commit -m "feat: add citation extraction with regex patterns"
```

---

### Task 4: Citation resolution — tests and implementation

**Files:**
- Modify: `tests/test_citations.py` (append new test class)
- Modify: `citations.py` (add resolution functions)

**Step 1: Write failing resolution tests**

Append to `tests/test_citations.py`:

```python
class TestCitationResolution:
    """Tests for resolving citations to corpus opinion IDs."""

    def test_build_citation_index_from_titles(self):
        from citations import build_citation_index
        opinions = [
            (1, "Smith v. Jones, 123 F.2d 456 (2d Cir. 1990)"),
            (2, "Doe v. Roe, 789 F.3d 101 (9th Cir. 2015)"),
            (3, "No citation in this title"),
        ]
        index = build_citation_index(opinions)
        assert index[("123", "F.2d", "456")] == 1
        assert index[("789", "F.3d", "101")] == 2
        assert len(index) == 2

    def test_resolve_citations_links_to_corpus(self):
        from citations import resolve_citations
        engine = get_local_engine()
        init_local_db(engine)
        session = get_session(engine)

        from sqlalchemy import text as sql_text
        session.execute(sql_text("DELETE FROM citations"))
        session.execute(sql_text("DELETE FROM opinions"))
        session.commit()

        # Opinion 1 cites opinion 2
        session.add(Opinion(id=1, package_id="pkg_1", title="Case A"))
        session.add(Opinion(id=2, package_id="pkg_2",
                            title="Case B, 500 F.3d 100 (3d Cir. 2007)"))
        session.commit()

        session.add(Citation(
            citing_opinion_id=1,
            volume="500", reporter="F.3d", page="100",
            citation_string="500 F.3d 100", context_snippet="context",
        ))
        session.commit()
        session.close()

        resolved = resolve_citations(engine)
        assert resolved > 0

        session = get_session(engine)
        cite = session.query(Citation).filter_by(citing_opinion_id=1).first()
        assert cite.cited_opinion_id == 2
        session.close()

    def test_unresolved_stays_null(self):
        from citations import resolve_citations
        engine = get_local_engine()
        init_local_db(engine)
        session = get_session(engine)

        from sqlalchemy import text as sql_text
        session.execute(sql_text("DELETE FROM citations"))
        session.execute(sql_text("DELETE FROM opinions"))
        session.commit()

        session.add(Opinion(id=1, package_id="pkg_1", title="Case A"))
        session.commit()

        # Citation to something not in corpus
        session.add(Citation(
            citing_opinion_id=1,
            volume="999", reporter="F.2d", page="888",
            citation_string="999 F.2d 888", context_snippet="context",
        ))
        session.commit()
        session.close()

        resolve_citations(engine)

        session = get_session(engine)
        cite = session.query(Citation).filter_by(citing_opinion_id=1).first()
        assert cite.cited_opinion_id is None
        session.close()
```

**Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_citations.py::TestCitationResolution -v`
Expected: FAIL

**Step 3: Implement resolution functions in citations.py**

Add to `citations.py`:

```python
def build_citation_index(opinions):
    """Build (volume, reporter, page) -> opinion_id index from opinion titles.

    Args:
        opinions: list of (opinion_id, title) tuples

    Returns:
        dict mapping (volume, reporter, page) -> opinion_id
    """
    index = {}
    for opinion_id, title in opinions:
        if not title:
            continue
        cites = extract_citations_from_text(title)
        for cite in cites:
            key = (cite["volume"], cite["reporter"], cite["page"])
            if key not in index:
                index[key] = opinion_id
    return index


def resolve_citations(engine):
    """Resolve citations to corpus opinion IDs using title-based index.

    Returns count of newly resolved citations.
    """
    session = get_session(engine)
    try:
        # Build index from all opinion titles
        opinions = session.execute(
            text("SELECT id, title FROM opinions WHERE title IS NOT NULL")
        ).fetchall()
        cite_index = build_citation_index(opinions)
        logger.info(f"Built citation index with {len(cite_index)} entries")

        # Resolve unresolved citations
        unresolved = session.query(Citation).filter(
            Citation.cited_opinion_id.is_(None)
        ).all()

        resolved_count = 0
        for cite in unresolved:
            key = (cite.volume, cite.reporter, cite.page)
            if key in cite_index:
                cite.cited_opinion_id = cite_index[key]
                resolved_count += 1

        session.commit()
        logger.info(f"Resolved {resolved_count}/{len(unresolved)} citations")
        return resolved_count
    finally:
        session.close()
```

**Step 4: Run resolution tests**

Run: `python -m pytest tests/test_citations.py::TestCitationResolution -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add citations.py tests/test_citations.py
git commit -m "feat: add citation resolution via title-based index"
```

---

### Task 5: Graph metrics — tests and implementation

**Files:**
- Modify: `tests/test_citations.py` (append new test class)
- Modify: `citations.py` (add graph metric functions)

**Step 1: Write failing graph metric tests**

Append to `tests/test_citations.py`:

```python
class TestGraphMetrics:
    """Tests for NetworkX graph metrics computation."""

    def test_build_graph_from_citations(self):
        from citations import build_citation_graph
        edges = [(1, 2), (1, 3), (2, 3), (4, 1)]
        G = build_citation_graph(edges)
        assert G.number_of_nodes() == 4
        assert G.number_of_edges() == 4
        assert G.is_directed()

    def test_compute_metrics(self):
        from citations import build_citation_graph, compute_graph_metrics
        edges = [(1, 2), (1, 3), (2, 3), (4, 1), (4, 2)]
        G = build_citation_graph(edges)
        metrics = compute_graph_metrics(G)
        # Node 2 is cited by 1 and 4, and node 3 is cited by 1 and 2
        assert metrics[2]["in_degree"] == 2
        assert metrics[3]["in_degree"] == 2
        assert metrics[1]["out_degree"] == 2
        # PageRank should exist for all nodes
        for node_id in [1, 2, 3, 4]:
            assert "pagerank" in metrics[node_id]
            assert "hub_score" in metrics[node_id]
            assert "authority_score" in metrics[node_id]
            assert "community_id" in metrics[node_id]

    def test_store_and_load_metrics(self):
        from citations import store_metrics
        engine = get_local_engine()
        init_local_db(engine)
        session = get_session(engine)

        from sqlalchemy import text as sql_text
        session.execute(sql_text("DELETE FROM opinion_metrics"))
        session.execute(sql_text("DELETE FROM opinions"))
        session.commit()

        for oid in [1, 2, 3]:
            session.add(Opinion(id=oid, package_id=f"pkg_{oid}", title=f"Case {oid}"))
        session.commit()
        session.close()

        metrics = {
            1: {"in_degree": 2, "out_degree": 1, "pagerank": 0.4,
                "hub_score": 0.1, "authority_score": 0.5, "community_id": 0},
            2: {"in_degree": 1, "out_degree": 2, "pagerank": 0.3,
                "hub_score": 0.6, "authority_score": 0.2, "community_id": 0},
            3: {"in_degree": 0, "out_degree": 0, "pagerank": 0.1,
                "hub_score": 0.0, "authority_score": 0.0, "community_id": 1},
        }
        store_metrics(engine, metrics)

        session = get_session(engine)
        from db import OpinionMetric
        m = session.query(OpinionMetric).filter_by(opinion_id=1).first()
        assert m.in_degree == 2
        assert abs(m.pagerank - 0.4) < 1e-6
        assert m.community_id == 0
        session.close()
```

**Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_citations.py::TestGraphMetrics -v`
Expected: FAIL

**Step 3: Implement graph functions in citations.py**

Add to `citations.py`:

```python
def build_citation_graph(edges):
    """Build a directed graph from (citing_id, cited_id) edge list.

    Args:
        edges: list of (source, target) tuples

    Returns:
        networkx.DiGraph
    """
    G = nx.DiGraph()
    G.add_edges_from(edges)
    return G


def compute_graph_metrics(G):
    """Compute per-node metrics on citation graph.

    Returns dict mapping node_id -> {in_degree, out_degree, pagerank,
    hub_score, authority_score, community_id}.
    """
    metrics = {}

    # Degree
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())

    # PageRank
    pagerank = nx.pagerank(G, max_iter=100)

    # HITS
    hubs, authorities = nx.hits(G, max_iter=100)

    # Communities (Louvain on undirected copy)
    G_undirected = G.to_undirected()
    if G_undirected.number_of_nodes() > 0:
        communities = nx.community.louvain_communities(G_undirected, seed=42)
        node_community = {}
        for i, comm in enumerate(communities):
            for node in comm:
                node_community[node] = i
    else:
        node_community = {}

    for node in G.nodes():
        metrics[node] = {
            "in_degree": in_deg.get(node, 0),
            "out_degree": out_deg.get(node, 0),
            "pagerank": pagerank.get(node, 0.0),
            "hub_score": hubs.get(node, 0.0),
            "authority_score": authorities.get(node, 0.0),
            "community_id": node_community.get(node, -1),
        }

    return metrics


def store_metrics(engine, metrics):
    """Store per-opinion graph metrics to opinion_metrics table.

    Clears existing rows, then inserts new ones.

    Args:
        engine: SQLAlchemy engine
        metrics: dict mapping opinion_id -> {in_degree, out_degree, pagerank, ...}
    """
    session = get_session(engine)
    try:
        session.execute(text("DELETE FROM opinion_metrics"))
        session.commit()

        for opinion_id, m in metrics.items():
            session.add(OpinionMetric(
                opinion_id=opinion_id,
                in_degree=m["in_degree"],
                out_degree=m["out_degree"],
                pagerank=m["pagerank"],
                hub_score=m["hub_score"],
                authority_score=m["authority_score"],
                community_id=m["community_id"],
            ))
        session.commit()
        logger.info(f"Stored metrics for {len(metrics)} opinions")
    finally:
        session.close()
```

**Step 4: Run graph metric tests**

Run: `python -m pytest tests/test_citations.py::TestGraphMetrics -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add citations.py tests/test_citations.py
git commit -m "feat: add citation graph metrics (PageRank, HITS, Louvain)"
```

---

### Task 6: Full pipeline function and CLI

**Files:**
- Modify: `citations.py` (add `run_citation_analysis()` and `main()`)

**Step 1: Add the full pipeline function and CLI to citations.py**

Add to `citations.py`:

```python
def run_citation_analysis(engine=None, extract_only=False, metrics_only=False):
    """Full citation analysis pipeline.

    1. Extract citations from all opinion plain_text
    2. Resolve to corpus opinion IDs
    3. Build graph and compute metrics
    4. Store metrics to DB
    """
    if engine is None:
        engine = get_local_engine()
        init_local_db(engine)

    if not metrics_only:
        # Step 1: Extract citations
        session = get_session(engine)
        opinions = session.execute(
            text("SELECT id, plain_text FROM opinions "
                 "WHERE plain_text IS NOT NULL AND plain_text != ''")
        ).fetchall()
        session.close()

        logger.info(f"Extracting citations from {len(opinions)} opinions...")
        total_cites = 0
        for i, (opinion_id, plain_text) in enumerate(opinions):
            cites = extract_citations_from_text(plain_text)
            if cites:
                stored = store_citations(engine, opinion_id, cites)
                total_cites += stored
            if (i + 1) % 5000 == 0:
                logger.info(f"  Progress: {i + 1}/{len(opinions)} opinions, "
                            f"{total_cites} citations stored")

        logger.info(f"Extraction complete: {total_cites} citations from "
                    f"{len(opinions)} opinions")

        # Step 2: Resolve
        resolved = resolve_citations(engine)
        logger.info(f"Resolved {resolved} citations to corpus opinion IDs")

        if extract_only:
            return

    # Step 3: Build graph from resolved citations
    session = get_session(engine)
    edges = session.execute(
        text("SELECT citing_opinion_id, cited_opinion_id FROM citations "
             "WHERE cited_opinion_id IS NOT NULL")
    ).fetchall()
    session.close()

    if not edges:
        logger.warning("No resolved citations found. Cannot compute metrics.")
        return

    edge_list = [(row[0], row[1]) for row in edges]
    G = build_citation_graph(edge_list)
    logger.info(f"Built graph: {G.number_of_nodes()} nodes, "
                f"{G.number_of_edges()} edges")

    # Step 4: Compute and store metrics
    metrics = compute_graph_metrics(G)
    store_metrics(engine, metrics)

    logger.info("Citation analysis pipeline complete.")


def get_citation_summary(engine):
    """Print citation summary stats."""
    with engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM citations")).scalar()
        resolved = conn.execute(
            text("SELECT COUNT(*) FROM citations WHERE cited_opinion_id IS NOT NULL")
        ).scalar()
        unique_cited = conn.execute(
            text("SELECT COUNT(DISTINCT cited_opinion_id) FROM citations "
                 "WHERE cited_opinion_id IS NOT NULL")
        ).scalar()
        opinions_with_metrics = conn.execute(
            text("SELECT COUNT(*) FROM opinion_metrics")
        ).scalar()

    return {
        "total_citations": total,
        "resolved": resolved,
        "unresolved": total - resolved,
        "resolution_rate": f"{resolved / total * 100:.1f}%" if total > 0 else "N/A",
        "unique_cases_cited": unique_cited,
        "opinions_with_metrics": opinions_with_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Citation network analysis")
    parser.add_argument("--extract-only", action="store_true",
                        help="Just extract citations, don't compute metrics")
    parser.add_argument("--metrics-only", action="store_true",
                        help="Recompute metrics from existing citations")
    parser.add_argument("--info", action="store_true",
                        help="Print citation summary")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    engine = get_local_engine()
    init_local_db(engine)

    if args.info:
        stats = get_citation_summary(engine)
        for key, value in stats.items():
            print(f"  {key}: {value}")
        return

    run_citation_analysis(engine, extract_only=args.extract_only,
                          metrics_only=args.metrics_only)


if __name__ == "__main__":
    main()
```

**Step 2: Run all citation tests**

Run: `python -m pytest tests/test_citations.py -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add citations.py
git commit -m "feat: add citation analysis CLI with extract/resolve/metrics pipeline"
```

---

### Task 7: Run extraction on real data

**Step 1: Run extraction only**

Run: `python citations.py --extract-only`
Expected: Extracts citations from ~30K opinions, logs progress every 5K. Takes ~5-10 minutes.

**Step 2: Check summary**

Run: `python citations.py --info`
Expected: Shows total citations, resolved count, resolution rate.

**Step 3: Run full pipeline (metrics)**

Run: `python citations.py --metrics-only`
Expected: Builds graph, computes PageRank/HITS/communities. Takes seconds.

**Step 4: Verify summary again**

Run: `python citations.py --info`
Expected: Shows opinions_with_metrics > 0.

**Step 5: Commit any fixes if needed**

---

### Task 8: Citations dashboard

**Files:**
- Modify: `pages/4_Citations.py` (replace placeholder)

**Step 1: Replace placeholder with full dashboard**

Replace `pages/4_Citations.py` with:

```python
import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import text

from db import get_local_engine, init_local_db, get_session

st.set_page_config(page_title="Citations", page_icon="⚖️", layout="wide")
st.title("Citation Network")


@st.cache_resource
def get_db_engine():
    engine = get_local_engine()
    init_local_db(engine)
    return engine


@st.cache_data
def load_citation_summary(_engine):
    with _engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM citations")).scalar()
        resolved = conn.execute(
            text("SELECT COUNT(*) FROM citations WHERE cited_opinion_id IS NOT NULL")
        ).scalar()
        unique_cited = conn.execute(
            text("SELECT COUNT(DISTINCT cited_opinion_id) FROM citations "
                 "WHERE cited_opinion_id IS NOT NULL")
        ).scalar()
        n_metrics = conn.execute(
            text("SELECT COUNT(*) FROM opinion_metrics")
        ).scalar()
    return {
        "total": total or 0,
        "resolved": resolved or 0,
        "unique_cited": unique_cited or 0,
        "n_metrics": n_metrics or 0,
    }


@st.cache_data
def load_most_cited(_engine, limit=20):
    """Top cited opinions by in-degree, with opinion metadata."""
    session = get_session(_engine)
    rows = session.execute(text(
        "SELECT m.opinion_id, m.in_degree, m.pagerank, m.community_id, "
        "o.title, o.court_type, o.circuit, o.date_issued "
        "FROM opinion_metrics m "
        "JOIN opinions o ON m.opinion_id = o.id "
        "ORDER BY m.in_degree DESC "
        "LIMIT :limit"
    ), {"limit": limit}).fetchall()
    session.close()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=[
        "opinion_id", "in_degree", "pagerank", "community_id",
        "title", "court_type", "circuit", "date_issued",
    ])


@st.cache_data
def load_most_influential(_engine, limit=20):
    """Top opinions by PageRank."""
    session = get_session(_engine)
    rows = session.execute(text(
        "SELECT m.opinion_id, m.pagerank, m.in_degree, m.hub_score, "
        "m.authority_score, m.community_id, "
        "o.title, o.court_type, o.circuit, o.date_issued "
        "FROM opinion_metrics m "
        "JOIN opinions o ON m.opinion_id = o.id "
        "ORDER BY m.pagerank DESC "
        "LIMIT :limit"
    ), {"limit": limit}).fetchall()
    session.close()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=[
        "opinion_id", "pagerank", "in_degree", "hub_score",
        "authority_score", "community_id",
        "title", "court_type", "circuit", "date_issued",
    ])


@st.cache_data
def load_community_sizes(_engine):
    """Community sizes from opinion_metrics."""
    session = get_session(_engine)
    rows = session.execute(text(
        "SELECT community_id, COUNT(*) as size "
        "FROM opinion_metrics "
        "WHERE community_id IS NOT NULL "
        "GROUP BY community_id "
        "ORDER BY size DESC"
    )).fetchall()
    session.close()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=["community_id", "size"])


@st.cache_data
def load_opinion_citations(_engine, opinion_id):
    """Load outgoing and incoming citations for one opinion."""
    session = get_session(_engine)

    # Outgoing (this opinion cites...)
    outgoing = session.execute(text(
        "SELECT c.citation_string, c.cited_opinion_id, c.context_snippet, "
        "o.title as cited_title "
        "FROM citations c "
        "LEFT JOIN opinions o ON c.cited_opinion_id = o.id "
        "WHERE c.citing_opinion_id = :oid"
    ), {"oid": opinion_id}).fetchall()

    # Incoming (cited by...)
    incoming = session.execute(text(
        "SELECT o.title, o.id as citing_id "
        "FROM citations c "
        "JOIN opinions o ON c.citing_opinion_id = o.id "
        "WHERE c.cited_opinion_id = :oid"
    ), {"oid": opinion_id}).fetchall()

    # Metrics
    metric = session.execute(text(
        "SELECT in_degree, out_degree, pagerank, hub_score, "
        "authority_score, community_id "
        "FROM opinion_metrics WHERE opinion_id = :oid"
    ), {"oid": opinion_id}).fetchone()

    session.close()
    return outgoing, incoming, metric


@st.cache_data
def load_opinion_statutes(_engine):
    """Map opinion_id -> statute key for filtering."""
    session = get_session(_engine)
    rows = session.execute(text(
        "SELECT os.opinion_id, UPPER(s.key) "
        "FROM opinion_statutes os "
        "JOIN statutes s ON os.statute_id = s.id"
    )).fetchall()
    session.close()
    return pd.DataFrame(rows, columns=["opinion_id", "statute"])


@st.cache_data
def load_opinions_for_select(_engine):
    """Load opinion IDs and titles for selectbox."""
    session = get_session(_engine)
    rows = session.execute(text(
        "SELECT o.id, o.title FROM opinions o "
        "JOIN opinion_metrics m ON o.id = m.opinion_id "
        "ORDER BY m.in_degree DESC"
    )).fetchall()
    session.close()
    return rows


engine = get_db_engine()
summary = load_citation_summary(engine)

if summary["total"] == 0:
    st.warning("No citations found. Run `python citations.py` to extract citations.")
    st.stop()

# --- Sidebar Filters ---
with st.sidebar:
    st.header("Filters")
    statute_options = ["All", "FDCPA", "TCPA", "FCRA"]
    selected_statute = st.selectbox("Statute", statute_options)

    df_statutes = load_opinion_statutes(engine)

# --- Summary Metrics ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Citations", f"{summary['total']:,}")
with col2:
    st.metric("Resolved (Internal)", f"{summary['resolved']:,}")
with col3:
    st.metric("Unique Cases Cited", f"{summary['unique_cited']:,}")
with col4:
    rate = (
        f"{summary['resolved'] / summary['total'] * 100:.1f}%"
        if summary["total"] > 0 else "N/A"
    )
    st.metric("Resolution Rate", rate)

# --- Most-Cited Cases ---
st.subheader("Most-Cited Cases (In-Degree)")
df_cited = load_most_cited(engine, limit=20)
if not df_cited.empty:
    display_cols = ["title", "court_type", "circuit", "date_issued", "in_degree", "pagerank"]
    df_display = df_cited[display_cols].copy()
    df_display["pagerank"] = df_display["pagerank"].apply(lambda x: f"{x:.6f}")
    st.dataframe(df_display, use_container_width=True, hide_index=True)
else:
    st.info("No metrics computed yet.")

# --- Most Influential (PageRank) ---
st.subheader("Most Influential Cases (PageRank)")
df_influential = load_most_influential(engine, limit=20)
if not df_influential.empty:
    display_cols = ["title", "court_type", "circuit", "pagerank", "in_degree", "authority_score"]
    df_display = df_influential[display_cols].copy()
    df_display["pagerank"] = df_display["pagerank"].apply(lambda x: f"{x:.6f}")
    df_display["authority_score"] = df_display["authority_score"].apply(lambda x: f"{x:.6f}")
    st.dataframe(df_display, use_container_width=True, hide_index=True)

# --- Citation Communities ---
st.subheader("Citation Communities")
df_communities = load_community_sizes(engine)
if not df_communities.empty:
    top_communities = df_communities.head(20)
    top_communities["label"] = top_communities["community_id"].apply(
        lambda c: f"Community {c}"
    )
    fig_comm = px.bar(
        top_communities, x="label", y="size",
        labels={"label": "Community", "size": "Number of Opinions"},
        color="size", color_continuous_scale="Blues",
    )
    fig_comm.update_layout(showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig_comm, use_container_width=True)

# --- Per-Opinion Drill-Down ---
st.subheader("Per-Opinion Drill-Down")

opinions_list = load_opinions_for_select(engine)
if opinions_list:
    options = {f"{row[0]} — {row[1][:80]}": row[0] for row in opinions_list}
    selected_label = st.selectbox("Select an opinion", list(options.keys()))
    selected_id = options[selected_label]

    outgoing, incoming, metric = load_opinion_citations(engine, selected_id)

    if metric:
        mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
        with mcol1:
            st.metric("In-Degree", metric[0])
        with mcol2:
            st.metric("Out-Degree", metric[1])
        with mcol3:
            st.metric("PageRank", f"{metric[2]:.6f}")
        with mcol4:
            st.metric("Hub Score", f"{metric[3]:.6f}")
        with mcol5:
            st.metric("Authority", f"{metric[4]:.6f}")

    col_out, col_in = st.columns(2)

    with col_out:
        st.markdown("**Cites (Outgoing)**")
        if outgoing:
            df_out = pd.DataFrame(outgoing, columns=[
                "citation_string", "cited_opinion_id", "context_snippet", "cited_title",
            ])
            df_out["cited_title"] = df_out["cited_title"].fillna("(external)")
            st.dataframe(
                df_out[["citation_string", "cited_title"]],
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("No outgoing citations.")

    with col_in:
        st.markdown("**Cited By (Incoming)**")
        if incoming:
            df_in = pd.DataFrame(incoming, columns=["title", "citing_id"])
            st.dataframe(df_in, use_container_width=True, hide_index=True)
        else:
            st.info("No incoming citations from corpus.")
```

**Step 2: Verify dashboard loads**

Run: `python -m streamlit run app.py --server.headless true`
Navigate to Citations page. Verify it either shows data or shows "Run python citations.py" message.

**Step 3: Commit**

```bash
git add pages/4_Citations.py
git commit -m "feat: add Citations dashboard with metrics and drill-down"
```

---

### Task 9: Pipeline integration

**Files:**
- Modify: `pipeline.py` (add `--citations` flag)

**Step 1: Add citations import and flag to pipeline.py**

Add import at top of `pipeline.py`:
```python
from citations import run_citation_analysis
```

Add `citations=False` parameter to `run_pipeline()` function signature.

Add at end of `run_pipeline()`, after the `topics` block:
```python
    if citations:
        logger.info("Running citation analysis...")
        run_citation_analysis(engine)
```

Add argparse argument:
```python
    parser.add_argument("--citations", action="store_true",
                        help="Run citation network analysis")
```

Pass to `run_pipeline()`:
```python
        citations=args.citations,
```

**Step 2: Run all tests**

Run: `python -m pytest -v`
Expected: All tests PASS (70+ existing + new citation tests)

**Step 3: Commit**

```bash
git add pipeline.py
git commit -m "feat: add --citations flag to pipeline"
```

---

### Task 10: Final validation

**Step 1: Run full test suite**

Run: `python -m pytest -v`
Expected: All PASS

**Step 2: Verify CLI works**

Run: `python citations.py --info`
Expected: Shows summary stats

**Step 3: Verify dashboard**

Run: `python -m streamlit run app.py --server.headless true`
Check Citations page loads with data.

**Step 4: Final commit if any fixes**

```bash
git add -A
git commit -m "fix: citation network final adjustments"
```
