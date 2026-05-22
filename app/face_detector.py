import io
import numpy as np
from PIL import Image


class FaceDetector:
    def __init__(self):
        self._model = None

    def _get_model(self):
        if self._model is None:
            from deepface import DeepFace
            self._model = DeepFace
        return self._model

    def detect_faces(self, image_bytes: bytes) -> list[np.ndarray]:
        model = self._get_model()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_array = np.array(img)
        try:
            results = model.represent(
                img_path=img_array,
                model_name="Facenet512",
                detector_backend="retinaface",
                enforce_detection=False,
                align=True,
            )
            embeddings = []
            for r in results:
                emb = np.array(r["embedding"], dtype=np.float32)
                # Only include if a face was actually detected (not blank region)
                if r.get("face_confidence", 1.0) > 0.7:
                    embeddings.append(emb)
            return embeddings
        except Exception:
            return []
