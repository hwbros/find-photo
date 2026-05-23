import io
import logging
import os

import numpy as np

log = logging.getLogger(__name__)

MIN_DET_SCORE = 0.5   # RetinaFace detection confidence
MIN_FACE_PX   = 40    # ignore faces smaller than 40px (likely noise)


def _setup_onnx_gpu() -> list[str]:
    """
    Register PyTorch's bundled CUDA DLLs so onnxruntime-gpu can find them,
    then return the best available providers.
    Must be called before onnxruntime is imported.
    """
    try:
        import torch
        from pathlib import Path
        torch_lib = Path(torch.__file__).parent / "lib"
        if torch_lib.exists():
            os.add_dll_directory(str(torch_lib))          # Windows DLL search path
            os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")
    except Exception as e:
        log.debug("CUDA DLL registration skipped: %s", e)

    import onnxruntime as ort
    available = ort.get_available_providers()
    log.info("onnxruntime providers available: %s", available)
    return (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available
        else ["CPUExecutionProvider"]
    )


def _decode_image(image_bytes: bytes) -> "np.ndarray | None":
    """Decode image bytes to a BGR numpy array (InsightFace expects BGR)."""
    from PIL import Image
    try:
        img_rgb = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
        return img_rgb[:, :, ::-1].copy()  # RGB → BGR, contiguous
    except Exception:
        return None


class FaceDetector:
    def __init__(self):
        self._app = None
        self._providers: list[str] | None = None

    def _load(self):
        if self._app is not None:
            return
        # Setup must happen before onnxruntime is imported elsewhere
        self._providers = _setup_onnx_gpu()
        log.info("InsightFace using providers: %s", self._providers)

        from insightface.app import FaceAnalysis
        self._app = FaceAnalysis(name="buffalo_l", providers=self._providers)
        self._app.prepare(ctx_id=0, det_size=(640, 640))
        log.info("InsightFace buffalo_l loaded (ArcFace + RetinaFace)")

    def detect_faces(self, image_bytes: bytes) -> list[np.ndarray]:
        """Return list of L2-normalised 512-d ArcFace embeddings."""
        self._load()
        try:
            img = _decode_image(image_bytes)
            if img is None:
                return []
            faces = self._app.get(img)
            result = []
            for face in faces:
                if face.det_score < MIN_DET_SCORE:
                    continue
                box = face.bbox  # [x1, y1, x2, y2]
                if (box[2] - box[0]) < MIN_FACE_PX or (box[3] - box[1]) < MIN_FACE_PX:
                    continue
                result.append(face.normed_embedding.astype(np.float32))
            return result
        except Exception as exc:
            log.warning("face detection error: %s", exc)
            return []
