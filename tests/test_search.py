import os
import pytest
import numpy as np

os.environ["ML_LOCAL_DB"] = ":memory:"


@pytest.fixture
def search_index():
    """Build a FAISS index with test data."""
    from embed import embed_chunks
    from index import build_index, add_to_index

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

    index = build_index()
    chunk_map = []
    add_to_index(index, chunk_map, chunks, embeddings)
    return index, chunk_map


def test_search_returns_results(search_index):
    from search import search_opinions
    index, chunk_map = search_index
    results = search_opinions(index, chunk_map, "debt collector called third party")
    assert len(results) > 0


def test_search_groups_by_opinion(search_index):
    from search import search_opinions
    index, chunk_map = search_index
    results = search_opinions(index, chunk_map, "debt collector damages", top_k=10)
    opinion_ids = [r["opinion_id"] for r in results]
    assert len(opinion_ids) == len(set(opinion_ids))


def test_search_best_passage_selected(search_index):
    from search import search_opinions
    index, chunk_map = search_index
    results = search_opinions(index, chunk_map, "robocall cellular telephone TCPA")
    assert results[0]["opinion_id"] == 2


def test_search_with_statute_filter(search_index):
    from search import search_opinions
    index, chunk_map = search_index
    results = search_opinions(
        index, chunk_map, "violation", filters={"statute": "TCPA"}
    )
    for r in results:
        assert "TCPA" in r["statutes"]
