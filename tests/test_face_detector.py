import io
import numpy as np
import torch
from PIL import Image
from unittest.mock import MagicMock
from app.face_detector import FaceDetector


def _jpeg(w=100, h=100) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color=(128, 64, 32)).save(buf, format="JPEG")
    return buf.getvalue()


IMG = _jpeg()


def make_detector(face_tensors, probs, embeddings):
    """
    face_tensors: list of (3,160,160) tensors or None
    probs:        list of floats or None
    embeddings:   list of (512,) tensors returned by resnet
    """
    det = FaceDetector()
    det._device = torch.device("cpu")

    mock_mtcnn = MagicMock()
    if face_tensors is None:
        mock_mtcnn.return_value = (None, None)
    else:
        stacked = torch.stack(face_tensors) if face_tensors else None
        mock_mtcnn.return_value = (stacked, probs)
    det._mtcnn = mock_mtcnn

    mock_resnet = MagicMock()
    mock_resnet.return_value = torch.stack(embeddings) if embeddings else torch.zeros(0, 512)
    det._resnet = mock_resnet

    return det


def _face_tensor():
    return torch.zeros(3, 160, 160)


def _emb():
    return torch.zeros(512)


def test_returns_embedding_for_detected_face():
    det = make_detector([_face_tensor()], [0.95], [_emb()])
    result = det.detect_faces(IMG)
    assert len(result) == 1
    assert isinstance(result[0], np.ndarray)
    assert result[0].shape == (512,)


def test_filters_low_confidence():
    det = make_detector([_face_tensor()], [0.5], [_emb()])
    result = det.detect_faces(IMG)
    assert result == []


def test_multiple_faces():
    det = make_detector(
        [_face_tensor(), _face_tensor()],
        [0.95, 0.92],
        [_emb(), _emb()],
    )
    result = det.detect_faces(IMG)
    assert len(result) == 2


def test_no_face_returns_empty():
    det = make_detector(None, None, [])
    result = det.detect_faces(IMG)
    assert result == []


def test_exception_returns_empty():
    det = FaceDetector()
    det._device = torch.device("cpu")
    mock_mtcnn = MagicMock(side_effect=Exception("crash"))
    det._mtcnn = mock_mtcnn
    det._resnet = MagicMock()
    result = det.detect_faces(IMG)
    assert result == []
