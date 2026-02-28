import os
import pytest
import numpy as np

os.environ["ML_LOCAL_DB"] = ":memory:"


def test_embed_chunks_returns_correct_shape():
    from embed import embed_chunks
    texts = ["This is sentence one.", "This is sentence two.", "Third sentence here."]
    embeddings = embed_chunks(texts)
    assert embeddings.shape == (3, 384)


def test_embed_chunks_normalized():
    from embed import embed_chunks
    texts = ["Test sentence for normalization."]
    embeddings = embed_chunks(texts)
    norm = np.linalg.norm(embeddings[0])
    assert abs(norm - 1.0) < 0.01


def test_faiss_index_add_and_search():
    from embed import embed_chunks
    from index import build_index, add_to_index

    chunks = [
        {"chunk_id": "1_chunk_0", "opinion_id": 1, "chunk_index": 0,
         "title": "Smith v. Collections", "court_name": "SD Texas",
         "court_type": "district", "circuit": "5th",
         "date_issued": "2024-01-15", "statutes": "FDCPA"},
        {"chunk_id": "2_chunk_0", "opinion_id": 2, "chunk_index": 0,
         "title": "Jones v. Telecom", "court_name": "ND Ohio",
         "court_type": "district", "circuit": "6th",
         "date_issued": "2024-03-20", "statutes": "TCPA"},
    ]
    texts = ["The debt collector called a third party and disclosed the debt.",
             "The plaintiff received an autodialed robocall on their cell phone."]
    embeddings = embed_chunks(texts)

    index = build_index()
    chunk_map = []
    add_to_index(index, chunk_map, chunks, embeddings)

    assert index.ntotal == 2
    assert len(chunk_map) == 2

    # Search for debt-related content
    query_emb = embed_chunks(["debt collector called third party"])
    if query_emb.dtype != np.float32:
        query_emb = query_emb.astype(np.float32)
    distances, indices = index.search(query_emb, 2)

    # First result should be the debt-related chunk
    assert chunk_map[indices[0][0]]["opinion_id"] == 1


def test_faiss_save_and_load(tmp_path):
    from embed import embed_chunks
    from index import build_index, add_to_index, save_index, load_index

    chunks = [
        {"chunk_id": "1_chunk_0", "opinion_id": 1, "chunk_index": 0,
         "statutes": "FDCPA"},
    ]
    texts = ["Test text for saving."]
    embeddings = embed_chunks(texts)

    index = build_index()
    chunk_map = []
    add_to_index(index, chunk_map, chunks, embeddings)

    idx_path = str(tmp_path / "test.index")
    map_path = str(tmp_path / "test.json")
    save_index(index, chunk_map, idx_path, map_path)

    loaded_index, loaded_map = load_index(idx_path, map_path)
    assert loaded_index.ntotal == 1
    assert loaded_map[0]["opinion_id"] == 1
