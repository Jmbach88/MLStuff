import logging
from typing import Optional

import nltk

logger = logging.getLogger(__name__)

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

        if current_len + sentence_len > chunk_size and current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(chunk_text)

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
