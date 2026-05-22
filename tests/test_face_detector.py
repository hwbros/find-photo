import io
import numpy as np
from PIL import Image
from unittest.mock import MagicMock
from app.face_detector import FaceDetector


def _jpeg(w=100, h=100) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color=(128, 64, 32)).save(buf, format="JPEG")
    return buf.getvalue()


IMG = _jpeg()


def make_detector(deepface_results):
    det = FaceDetector()
    mock_df = MagicMock()
    mock_df.represent.return_value = deepface_results
    det._model = mock_df
    return det


def test_returns_embedding_for_detected_face():
    det = make_detector([{"embedding": [0.1] * 512, "face_confidence": 0.95}])
    result = det.detect_faces(IMG)
    assert len(result) == 1
    assert isinstance(result[0], np.ndarray)
    assert result[0].shape == (512,)


def test_filters_low_confidence():
    det = make_detector([{"embedding": [0.1] * 512, "face_confidence": 0.5}])
    result = det.detect_faces(IMG)
    assert result == []


def test_multiple_faces():
    det = make_detector([
        {"embedding": [0.1] * 512, "face_confidence": 0.92},
        {"embedding": [0.2] * 512, "face_confidence": 0.88},
    ])
    result = det.detect_faces(IMG)
    assert len(result) == 2


def test_no_face_returns_empty():
    det = make_detector([])
    result = det.detect_faces(IMG)
    assert result == []


def test_exception_returns_empty():
    det = FaceDetector()
    mock_df = MagicMock()
    mock_df.represent.side_effect = Exception("model error")
    det._model = mock_df
    result = det.detect_faces(IMG)
    assert result == []
