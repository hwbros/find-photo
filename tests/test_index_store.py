import os
import pytest
from app.index_store import IndexStore


@pytest.fixture
def store(tmp_path):
    return IndexStore(db_path=str(tmp_path / "test.db"))


def test_not_indexed_initially(store):
    assert not store.is_indexed("folder1")


def test_init_and_complete(store):
    store.init_folder("f1", "https://drive.google.com/drive/folders/f1", 10)
    assert not store.is_indexed("f1")
    store.mark_complete("f1")
    assert store.is_indexed("f1")


def test_save_and_search_by_bib(store):
    store.init_folder("f1", "http://example.com", 1)
    store.save_photo("f1", "p1", "photo1.jpg", "http://drive/p1", ["1234", "5678"])
    store.mark_complete("f1")

    results = store.search_by_bib("f1", "1234")
    assert len(results) == 1
    assert results[0]["photo_id"] == "p1"


def test_no_substring_match(store):
    store.init_folder("f1", "http://example.com", 1)
    store.save_photo("f1", "p1", "photo1.jpg", "http://drive/p1", ["12345"])
    store.mark_complete("f1")

    assert store.search_by_bib("f1", "123") == []
    assert store.search_by_bib("f1", "12345") != []


def test_clear_folder(store):
    store.init_folder("f1", "http://example.com", 1)
    store.save_photo("f1", "p1", "photo1.jpg", "http://drive/p1", ["1234"])
    store.mark_complete("f1")
    store.clear_folder("f1")

    assert not store.is_indexed("f1")
    assert store.search_by_bib("f1", "1234") == []


def test_is_photo_indexed(store):
    store.init_folder("f1", "http://example.com", 2)
    assert not store.is_photo_indexed("f1", "p1")
    store.save_photo("f1", "p1", "photo1.jpg", "http://drive/p1", [])
    assert store.is_photo_indexed("f1", "p1")
