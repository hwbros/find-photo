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


def _normed_emb() -> np.ndarray:
    v = np.random.randn(512).astype(np.float32)
    return v / np.linalg.norm(v)


def _mock_face(det_score=0.99, bbox=(10, 10, 80, 80), emb=None):
    face = MagicMock()
    face.det_score = det_score
    face.bbox = np.array(bbox, dtype=np.float32)
    face.normed_embedding = emb if emb is not None else _normed_emb()
    return face


def make_detector(faces):
    """FaceDetector with _app pre-injected so _load() is never called."""
    det = FaceDetector()
    mock_app = MagicMock()
    mock_app.get.return_value = faces
    det._app = mock_app
    return det


def test_returns_embedding_for_detected_face():
    emb = _normed_emb()
    det = make_detector([_mock_face(emb=emb)])
    result = det.detect_faces(IMG)
    assert len(result) == 1
    assert isinstance(result[0], np.ndarray)
    assert result[0].shape == (512,)
    np.testing.assert_array_almost_equal(result[0], emb)


def test_filters_low_det_score():
    det = make_detector([_mock_face(det_score=0.3)])
    result = det.detect_faces(IMG)
    assert result == []


def test_filters_tiny_face():
    # face smaller than MIN_FACE_PX (40px) → ignored
    det = make_detector([_mock_face(bbox=(10, 10, 30, 30))])  # 20x20
    result = det.detect_faces(IMG)
    assert result == []


def test_multiple_faces():
    det = make_detector([_mock_face(), _mock_face()])
    result = det.detect_faces(IMG)
    assert len(result) == 2


def test_no_face_returns_empty():
    det = make_detector([])
    result = det.detect_faces(IMG)
    assert result == []


def test_exception_returns_empty():
    det = FaceDetector()
    mock_app = MagicMock()
    mock_app.get.side_effect = Exception("crash")
    det._app = mock_app
    result = det.detect_faces(IMG)
    assert result == []
