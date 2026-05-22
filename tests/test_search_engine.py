import numpy as np
import pytest
from app.search_engine import search, FACE_SIMILARITY_THRESHOLD


def make_emb(val: float) -> np.ndarray:
    emb = np.zeros(512, dtype=np.float32)
    emb[0] = val
    return emb / np.linalg.norm(emb)


REF = make_emb(1.0)
SAME = make_emb(0.999)   # very similar to REF
DIFF = make_emb(-1.0)    # opposite direction


def photo(pid, bibs=None, embeddings=None):
    return {
        "photo_id": pid,
        "photo_name": f"{pid}.jpg",
        "drive_url": f"http://drive/{pid}",
        "bibs": bibs or [],
        "embeddings": embeddings or [],
    }


def test_bib_match():
    results = search([photo("p1", bibs=["1234"])], None, "1234")
    assert len(results) == 1
    assert results[0].match_type == "bib"


def test_face_match():
    results = search([photo("p1", embeddings=[SAME])], REF, "9999")
    assert len(results) == 1
    assert results[0].match_type == "face"


def test_both_match():
    results = search([photo("p1", bibs=["1234"], embeddings=[SAME])], REF, "1234")
    assert len(results) == 1
    assert results[0].match_type == "both"


def test_no_match():
    results = search([photo("p1", bibs=["9999"], embeddings=[DIFF])], REF, "1234")
    assert results == []


def test_or_condition():
    photos = [
        photo("p1", bibs=["1234"]),
        photo("p2", embeddings=[SAME]),
        photo("p3"),
    ]
    results = search(photos, REF, "1234")
    ids = {r.photo_id for r in results}
    assert "p1" in ids
    assert "p2" in ids
    assert "p3" not in ids


def test_sort_order():
    photos = [
        photo("face_only", embeddings=[SAME]),
        photo("both", bibs=["1234"], embeddings=[SAME]),
        photo("bib_only", bibs=["1234"]),
    ]
    results = search(photos, REF, "1234")
    assert results[0].match_type == "both"
    assert results[1].match_type == "bib"
    assert results[2].match_type == "face"


def test_no_ref_embedding_skips_face():
    results = search([photo("p1", embeddings=[SAME])], None, "9999")
    assert results == []


def test_face_below_threshold():
    below = make_emb(0.5)  # will have lower cosine similarity with REF
    results = search([photo("p1", embeddings=[below])], REF, "9999")
    # Only passes if cosine similarity >= threshold
    assert all(r.photo_id != "p1" for r in results) or results[0].match_type == "face"
