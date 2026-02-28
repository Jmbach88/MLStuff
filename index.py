# index.py
import json
import logging
import os
from pathlib import Path

import faiss
import numpy as np

import config

logger = logging.getLogger(__name__)


def build_index(dim: int = None) -> faiss.IndexFlatIP:
    """Create a new FAISS index for inner product (cosine similarity with normalized vectors)."""
    if dim is None:
        dim = config.EMBEDDING_DIM
    index = faiss.IndexFlatIP(dim)
    return index


def save_index(index: faiss.IndexFlatIP, chunk_map: list[dict],
               index_path: str = None, map_path: str = None):
    """Save FAISS index and chunk metadata map to disk."""
    if index_path is None:
        index_path = config.FAISS_INDEX
    if map_path is None:
        map_path = config.FAISS_MAP

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)

    with open(map_path, "w") as f:
        json.dump(chunk_map, f)

    logger.info(f"Saved FAISS index ({index.ntotal} vectors) to {index_path}")


def load_index(index_path: str = None, map_path: str = None):
    """Load FAISS index and chunk metadata map from disk.
    Returns (index, chunk_map) or (None, None) if files don't exist.
    """
    if index_path is None:
        index_path = config.FAISS_INDEX
    if map_path is None:
        map_path = config.FAISS_MAP

    if not os.path.exists(index_path) or not os.path.exists(map_path):
        return None, None

    index = faiss.read_index(index_path)
    with open(map_path, "r") as f:
        chunk_map = json.load(f)

    logger.info(f"Loaded FAISS index ({index.ntotal} vectors)")
    return index, chunk_map


def add_to_index(index: faiss.IndexFlatIP, chunk_map: list[dict],
                 chunks: list[dict], embeddings: np.ndarray):
    """Add chunks and embeddings to a FAISS index.

    chunk_map is a list of metadata dicts, one per vector, in the same order
    as the vectors in the FAISS index. Each entry has:
        chunk_id, opinion_id, chunk_index, title, court_name,
        court_type, circuit, date_issued, statutes
    """
    # FAISS requires float32
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    index.add(embeddings)

    for c in chunks:
        chunk_map.append({
            "chunk_id": c["chunk_id"],
            "opinion_id": c["opinion_id"],
            "chunk_index": c["chunk_index"],
            "title": c.get("title", ""),
            "court_name": c.get("court_name", ""),
            "court_type": c.get("court_type", ""),
            "circuit": c.get("circuit", ""),
            "date_issued": c.get("date_issued", ""),
            "statutes": c.get("statutes", ""),
        })

    logger.info(f"Added {len(chunks)} vectors (total: {index.ntotal})")
