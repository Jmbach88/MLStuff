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

import networkx as nx
from sqlalchemy import text

from db import (
    get_local_engine, init_local_db, get_session,
    Opinion, Citation, OpinionMetric,
)

logger = logging.getLogger(__name__)

# Regex patterns for federal case citations.
# Order matters: longer/more-specific reporters must come before shorter ones.
CITATION_PATTERNS = [
    (r'(\d+)\s+F\.\s*Supp\.\s*3d\s+(\d+)', 'F. Supp. 3d'),
    (r'(\d+)\s+F\.\s*Supp\.\s*2d\s+(\d+)', 'F. Supp. 2d'),
    (r'(\d+)\s+F\.\s*Supp\.\s+(\d+)(?!\s*d\b)', 'F. Supp.'),
    (r'(\d+)\s+F\.4th\s+(\d+)', 'F.4th'),
    (r'(\d+)\s+F\.3d\s+(\d+)', 'F.3d'),
    (r'(\d+)\s+F\.2d\s+(\d+)', 'F.2d'),
    (r'(\d+)\s+F\.\s+(\d+)', 'F.'),
    (r'(\d+)\s+U\.S\.\s+(\d+)', 'U.S.'),
    (r'(\d+)\s+S\.\s?Ct\.\s+(\d+)', 'S. Ct.'),
    (r'(\d{4})\s+WL\s+(\d+)', 'WL'),
    (r'(\d{4})\s+U\.S\.\s+Dist\.\s+LEXIS\s+(\d+)', 'U.S. Dist. LEXIS'),
    (r'(\d{4})\s+U\.S\.\s+App\.\s+LEXIS\s+(\d+)', 'U.S. App. LEXIS'),
]

_COMPILED_PATTERNS = [(re.compile(p), r) for p, r in CITATION_PATTERNS]

CONTEXT_CHARS = 100


def extract_citations_from_text(text_content):
    """Extract federal case citations from plain text.

    Returns list of dicts with keys: volume, reporter, page, citation_string, context_snippet.
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


def build_citation_index(opinions):
    """Build (volume, reporter, page) -> opinion_id index from opinion titles."""
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
    """Resolve citations to corpus opinion IDs using title-based index."""
    session = get_session(engine)
    try:
        opinions = session.execute(
            text("SELECT id, title FROM opinions WHERE title IS NOT NULL")
        ).fetchall()
        cite_index = build_citation_index(opinions)
        logger.info(f"Built citation index with {len(cite_index)} entries")

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


def build_citation_graph(edges):
    """Build a directed graph from (citing_id, cited_id) edge list."""
    G = nx.DiGraph()
    G.add_edges_from(edges)
    return G


def compute_graph_metrics(G):
    """Compute per-node metrics on citation graph.

    Returns dict mapping node_id -> {in_degree, out_degree, pagerank,
    hub_score, authority_score, community_id}.
    """
    metrics = {}
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    pagerank = nx.pagerank(G, max_iter=100)
    hubs, authorities = nx.hits(G, max_iter=100)

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
    """Store per-opinion graph metrics. Clears existing rows first."""
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


def run_citation_analysis(engine=None, extract_only=False, metrics_only=False):
    """Full citation analysis pipeline."""
    if engine is None:
        engine = get_local_engine()
        init_local_db(engine)

    if not metrics_only:
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

        resolved = resolve_citations(engine)
        logger.info(f"Resolved {resolved} citations to corpus opinion IDs")

        if extract_only:
            return

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

    metrics = compute_graph_metrics(G)
    store_metrics(engine, metrics)
    logger.info("Citation analysis pipeline complete.")


def get_citation_summary(engine):
    """Get citation summary stats."""
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
