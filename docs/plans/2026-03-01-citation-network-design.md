# Citation Network Analysis — Design Document

## Goal

Extract case citations from 30K federal court opinions, resolve them to corpus opinion IDs where possible, build a directed citation graph, compute network metrics (PageRank, in-degree, HITS, communities), and visualize in a Citations dashboard with per-opinion drill-down.

## Citation Extraction

Regex-based extraction from `plain_text` targeting federal case citations:

**Federal Reporters:**
- `\d+ F\. \d+` — F. (first series)
- `\d+ F\.2d \d+` — F.2d
- `\d+ F\.3d \d+` — F.3d
- `\d+ F\.4th \d+` — F.4th
- `\d+ F\. Supp\. \d+` — F. Supp.
- `\d+ F\. Supp\. 2d \d+` — F. Supp. 2d
- `\d+ F\. Supp\. 3d \d+` — F. Supp. 3d

**US Reports / Supreme Court:**
- `\d+ U\.S\. \d+` — US Reports
- `\d+ S\. ?Ct\. \d+` — Supreme Court Reporter

**Westlaw:**
- `\d{4} WL \d+` — e.g., "2024 WL 1234567"

**LexisNexis:**
- `\d{4} U\.S\. Dist\. LEXIS \d+`
- `\d{4} U\.S\. App\. LEXIS \d+`

Each extraction captures volume, reporter, page (or year + ID for WL/LEXIS), plus ~100 chars of surrounding context. Deduplicate by (citing_opinion_id, volume, reporter, page) — store first occurrence only.

Scope: Federal case-to-case citations only. Skip state reporters, law reviews, statutory citations.

## Citation Resolution

Two-pass best-effort matching to link citations to corpus opinion IDs:

**Pass 1: Volume/Reporter/Page lookup.** Parse opinion titles to extract embedded citations, build index of (volume, reporter, page) -> opinion_id. O(1) lookups per extracted citation.

**Pass 2: Westlaw/LEXIS ID matching.** Match WL and LEXIS identifiers against opinion text headers.

Unresolved citations stored with `cited_opinion_id=NULL`. Expected resolution rate: 5-15% (most citations point outside any single corpus). Even low resolution yields thousands of internal edges.

## Graph Metrics

NetworkX directed graph. Nodes = opinion IDs, edges = resolved internal citations.

Per-opinion metrics:
- **In-degree** — times cited by corpus opinions
- **Out-degree** — number of corpus opinions cited
- **PageRank** — importance weighted by citer importance
- **Hub score** (HITS) — survey/overview quality
- **Authority score** (HITS) — authoritative standing
- **Community ID** (Louvain) — cluster of densely connected opinions (undirected)

## Database Schema

```
Citation:
    id (PK, autoincrement)
    citing_opinion_id (FK opinions.id)
    cited_opinion_id (FK opinions.id, nullable)
    volume (Text)
    reporter (Text)
    page (Text)
    citation_string (Text)
    context_snippet (Text)

OpinionMetric:
    opinion_id (PK, FK opinions.id)
    in_degree (Integer)
    out_degree (Integer)
    pagerank (Float)
    hub_score (Float)
    authority_score (Float)
    community_id (Integer)
```

## Citations Dashboard UI

Sidebar filters: Statute, Circuit, Court Type.

**Dashboard overview:**
1. Summary metrics — total citations, internal resolved, unique cases cited, graph density
2. Most-Cited Cases (In-Degree) — top 20 table with title, court, circuit, date, times cited
3. Most Influential Cases (PageRank) — top 20 by PageRank with in-degree for comparison
4. Citation Communities — bar chart of community sizes, top-cited opinion per community as label

**Per-opinion drill-down:**
- Selectbox to pick an opinion
- Cites (outgoing), Cited By (incoming), Metrics, Top external citations

## New/Modified Files

| File | Change |
|------|--------|
| `citations.py` | New: extraction, resolution, graph metrics, CLI |
| `tests/test_citations.py` | New: regex, dedup, resolution, graph metric tests |
| `pages/4_Citations.py` | Replace placeholder with dashboard |
| `db.py` | Add Citation and OpinionMetric ORM classes |
| `pipeline.py` | Add `--citations` flag |
| `requirements.txt` | Add `networkx>=3.0` |

## CLI

```
python citations.py                # extract + resolve + metrics
python citations.py --extract-only # just extract to DB
python citations.py --metrics-only # recompute from existing citations
python citations.py --info         # print summary
```

## Dependencies

- `networkx>=3.0`

## Expected Runtime

- Extraction: ~5-10 minutes (regex on 30K full texts)
- Resolution: seconds (dict lookups)
- Graph metrics: seconds (NetworkX on ~30K nodes)
