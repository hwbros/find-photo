import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)

MIN_DET_SCORE = 0.5   # RetinaFace detection confidence
MIN_FACE_PX   = 40    # ignore faces smaller than 40px (likely noise)


class FaceDetector:
    def __init__(self):
        self._app = None

    def _load(self):
        if self._app is not None:
            return
        from insightface.app import FaceAnalysis
        self._app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=0, det_size=(640, 640))
        log.info("InsightFace buffalo_l loaded (ArcFace + RetinaFace)")

    def detect_faces(self, image_bytes: bytes) -> list[np.ndarray]:
        """Return list of L2-normalised 512-d ArcFace embeddings."""
        self._load()
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # BGR, as InsightFace expects
            if img is None:
                return []
            faces = self._app.get(img)
            result = []
            for face in faces:
                if face.det_score < MIN_DET_SCORE:
                    continue
                box = face.bbox  # [x1, y1, x2, y2]
                w = box[2] - box[0]
                h = box[3] - box[1]
                if w < MIN_FACE_PX or h < MIN_FACE_PX:
                    continue
                result.append(face.normed_embedding.astype(np.float32))
            return result
        except Exception as exc:
            log.warning("face detection error: %s", exc)
            return []
