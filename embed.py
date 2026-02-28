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
