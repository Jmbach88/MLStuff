# search.py
import argparse
import logging

import numpy as np

from embed import embed_chunks, get_model
from index import load_index
import config

logger = logging.getLogger(__name__)


def search_opinions(
    index,
    chunk_map: list[dict],
    query: str,
    top_k: int = None,
    filters: dict = None,
) -> list[dict]:
    """Search for opinions matching the query.

    Args:
        index: FAISS index
        chunk_map: list of metadata dicts (one per vector, aligned with FAISS index)
        query: natural language search query
        top_k: number of results to return
        filters: optional dict with keys: statute, circuit, court_type, date_from, date_to, opinion_ids

    Returns a list of dicts, one per opinion, sorted by best match:
        opinion_id, title, court_name, circuit, date_issued,
        statutes, similarity_score, best_passage, chunk_index
    """
    if top_k is None:
        top_k = config.DEFAULT_TOP_K

    # Embed query
    query_emb = embed_chunks([query])
    if query_emb.dtype != np.float32:
        query_emb = query_emb.astype(np.float32)

    # Search FAISS — oversample to have enough after filtering
    n_search = min(top_k * config.OVERSAMPLE_FACTOR * 5, index.ntotal)
    if n_search == 0:
        return []

    distances, indices = index.search(query_emb, n_search)

    # Build results, apply filters, group by opinion
    opinion_best = {}

    for i, idx in enumerate(indices[0]):
        if idx == -1:  # FAISS returns -1 for empty slots
            continue

        meta = chunk_map[idx]
        similarity = float(distances[0][i])  # inner product = cosine sim for normalized vectors

        # Apply filters
        if not _passes_filters(meta, filters):
            continue

        opinion_id = meta["opinion_id"]

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
                "best_passage": meta.get("text", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "chunk_id": meta.get("chunk_id", ""),
            }

    # Sort by similarity descending, take top_k
    ranked = sorted(opinion_best.values(), key=lambda x: x["similarity_score"], reverse=True)
    return ranked[:top_k]


def _passes_filters(meta: dict, filters: dict) -> bool:
    """Check if a chunk's metadata passes all active filters."""
    if not filters:
        return True

    if filters.get("statute"):
        if filters["statute"] not in meta.get("statutes", ""):
            return False

    if filters.get("circuit"):
        if meta.get("circuit") != filters["circuit"]:
            return False

    if filters.get("court_type"):
        if meta.get("court_type") != filters["court_type"]:
            return False

    if filters.get("date_from"):
        if meta.get("date_issued", "") < filters["date_from"]:
            return False

    if filters.get("date_to"):
        if meta.get("date_issued", "") > filters["date_to"]:
            return False

    if filters.get("opinion_ids"):
        if meta.get("opinion_id") not in filters["opinion_ids"]:
            return False

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search federal court opinions")
    parser.add_argument("query", help="Natural language search query")
    parser.add_argument("--statute", help="Filter by statute (FDCPA, TCPA, FCRA)")
    parser.add_argument("--circuit", help="Filter by circuit (e.g., 5th)")
    parser.add_argument("--top", type=int, default=10, help="Number of results")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    index, chunk_map = load_index()
    if index is None:
        print("No index found. Run pipeline.py first.")
        exit(1)

    filters = {}
    if args.statute:
        filters["statute"] = args.statute
    if args.circuit:
        filters["circuit"] = args.circuit

    results = search_opinions(index, chunk_map, args.query, top_k=args.top, filters=filters)

    for i, r in enumerate(results, 1):
        print(f"\n--- Result {i} (score: {r['similarity_score']:.3f}) ---")
        print(f"  {r['title']}")
        print(f"  {r['court_name']} | {r['circuit']} Circuit | {r['date_issued']}")
        print(f"  Statutes: {r['statutes']}")
