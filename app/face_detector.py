import io

import numpy as np
import torch
from PIL import Image


class FaceDetector:
    def __init__(self):
        self._mtcnn = None
        self._resnet = None
        self._device = None

    def _load(self):
        if self._mtcnn is not None:
            return
        from facenet_pytorch import MTCNN, InceptionResnetV1
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._mtcnn = MTCNN(keep_all=True, device=self._device, post_process=True)
        self._resnet = InceptionResnetV1(pretrained="vggface2").eval().to(self._device)

    def detect_faces(self, image_bytes: bytes) -> list[np.ndarray]:
        self._load()
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            faces, probs = self._mtcnn(img, return_prob=True)

            if faces is None or probs is None:
                return []

            # keep_all=True always returns (n,3,160,160) even for n=1
            confident = [(f, p) for f, p in zip(faces, probs) if p > 0.80]
            if not confident:
                return []

            batch = torch.stack([f for f, _ in confident]).to(self._device)
            with torch.no_grad():
                embeddings = self._resnet(batch)

            return [emb.cpu().numpy() for emb in embeddings]
        except Exception:
            return []
