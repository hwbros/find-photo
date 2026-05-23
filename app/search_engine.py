from dataclasses import dataclass
from typing import Literal

import numpy as np

FACE_SIMILARITY_THRESHOLD = 0.35  # ArcFace (InsightFace buffalo_l) cosine similarity
# ArcFace score distribution: same-person ~0.4-0.8, different-person ~-0.2-0.3
# Threshold 0.35 gives ~0.01% false-accept rate on standard benchmarks


@dataclass
class MatchResult:
    photo_id: str
    photo_name: str
    drive_url: str
    match_type: Literal["bib", "face", "both"]
    face_score: float | None = None


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def search(
    indexed_photos: list[dict],
    ref_embedding: np.ndarray | None,
    bib_number: str,
) -> list[MatchResult]:
    results = []
    for photo in indexed_photos:
        bibs: list[str] = photo.get("bibs", [])
        embeddings: list[np.ndarray] = photo.get("embeddings", [])

        bib_match = bib_number.strip() in bibs

        # Accept pre-computed face score (from index) or compute on-the-fly
        face_score: float | None = photo.get("_face_score")
        face_match = face_score is not None and face_score >= FACE_SIMILARITY_THRESHOLD

        if not face_match and ref_embedding is not None and embeddings:
            scores = [_cosine_similarity(ref_embedding, emb) for emb in embeddings]
            best = max(scores)
            if best >= FACE_SIMILARITY_THRESHOLD:
                face_match = True
                face_score = best

        if not bib_match and not face_match:
            continue

        if bib_match and face_match:
            match_type = "both"
        elif bib_match:
            match_type = "bib"
        else:
            match_type = "face"

        results.append(MatchResult(
            photo_id=photo["photo_id"],
            photo_name=photo["photo_name"],
            drive_url=photo["drive_url"],
            match_type=match_type,
            face_score=round(face_score, 3) if face_score else None,
        ))

    # Sort: both > bib > face, then by face_score desc
    order = {"both": 0, "bib": 1, "face": 2}
    results.sort(key=lambda r: (order[r.match_type], -(r.face_score or 0)))
    return results
