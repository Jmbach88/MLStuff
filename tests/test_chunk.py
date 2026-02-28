import pytest
from chunk import chunk_opinion


def test_short_opinion_produces_single_chunk():
    text = "This is a short opinion. It has two sentences."
    chunks = chunk_opinion(opinion_id=1, text=text, chunk_size=2000, overlap=400)
    assert len(chunks) == 1
    assert chunks[0]["opinion_id"] == 1
    assert chunks[0]["chunk_id"] == "1_chunk_0"
    assert chunks[0]["chunk_index"] == 0
    assert chunks[0]["text"] == text


def test_long_opinion_produces_overlapping_chunks():
    sentences = [f"Sentence number {i} is here for padding purposes. " for i in range(80)]
    text = "".join(sentences)
    assert len(text) > 3000

    chunks = chunk_opinion(opinion_id=42, text=text, chunk_size=2000, overlap=400)
    assert len(chunks) >= 2

    end_of_first = chunks[0]["text"][-200:]
    assert end_of_first in chunks[1]["text"]

    for i, c in enumerate(chunks):
        assert c["chunk_index"] == i
        assert c["chunk_id"] == f"42_chunk_{i}"


def test_empty_text_returns_no_chunks():
    assert chunk_opinion(1, "", 2000, 400) == []
    assert chunk_opinion(1, "   ", 2000, 400) == []
    assert chunk_opinion(1, None, 2000, 400) == []


def test_chunks_never_split_mid_sentence():
    sentences = [f"Sentence {i} has some content here. " for i in range(100)]
    text = "".join(sentences)
    chunks = chunk_opinion(opinion_id=1, text=text, chunk_size=500, overlap=100)
    for c in chunks:
        stripped = c["text"].rstrip()
        assert stripped.endswith("."), f"Chunk does not end at sentence boundary: ...{stripped[-50:]}"
