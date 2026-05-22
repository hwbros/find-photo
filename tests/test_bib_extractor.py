import io
import pytest
from PIL import Image
from unittest.mock import MagicMock, patch
from app.bib_extractor import BibExtractor


def _jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (100, 100), color=(200, 200, 200)).save(buf, format="JPEG")
    return buf.getvalue()


IMG = _jpeg()


def make_extractor(ocr_results):
    """Return a BibExtractor whose EasyOCR reader is mocked."""
    extractor = BibExtractor()
    mock_reader = MagicMock()
    mock_reader.readtext.return_value = ocr_results
    extractor._reader = mock_reader
    return extractor


def test_returns_numeric_bib():
    ex = make_extractor([
        (None, "1234", 0.95),
        (None, "FINISH", 0.90),
    ])
    assert ex.extract_bibs(IMG) == ["1234"]


def test_ignores_low_confidence():
    ex = make_extractor([(None, "5678", 0.29)])
    assert ex.extract_bibs(IMG) == []


def test_ignores_non_numeric():
    ex = make_extractor([(None, "ABC", 0.99), (None, "12X4", 0.99)])
    assert ex.extract_bibs(IMG) == []


def test_ignores_single_digit():
    ex = make_extractor([(None, "7", 0.99)])
    assert ex.extract_bibs(IMG) == []


def test_strips_whitespace_in_text():
    ex = make_extractor([(None, "12 34", 0.99)])
    assert ex.extract_bibs(IMG) == ["1234"]


def test_multiple_bibs():
    ex = make_extractor([
        (None, "100", 0.99),
        (None, "200", 0.99),
    ])
    result = ex.extract_bibs(IMG)
    assert set(result) == {"100", "200"}


def test_empty_image():
    ex = make_extractor([])
    assert ex.extract_bibs(IMG) == []
