import json
from pathlib import Path

import cv2
import numpy as np

from hailo_detectord.config import Settings


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def compute_embedding(image: np.ndarray) -> list[float]:
    """Compute a deterministic test embedding for independent API validation.

    This is intentionally lightweight and dependency-free. It is not a production
    face embedding model. Replace this with a Hailo/ArcFace/InsightFace backend
    once the API contract is validated.
    """
    resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32).flatten()[::16]

    vector = np.concatenate([hist_h, hist_s, hist_v, gray])
    vector = _normalize(vector.astype(np.float32))

    return vector.tolist()


def cosine_similarity(left: list[float], right: list[float]) -> float:
    left_vector = np.asarray(left, dtype=np.float32)
    right_vector = np.asarray(right, dtype=np.float32)

    if left_vector.shape != right_vector.shape:
        return 0.0

    left_vector = _normalize(left_vector)
    right_vector = _normalize(right_vector)

    return float(np.dot(left_vector, right_vector))


class FaceLibrary:
    def __init__(self, settings: Settings) -> None:
        self.path = Path(settings.face_library_path or "/tmp/hailo-detectord-faces.json")
        self.threshold = settings.face_match_threshold
        self.records = self._load()

    def _load(self) -> dict[str, list[list[float]]]:
        if not self.path.exists():
            return {}

        data = json.loads(self.path.read_text(encoding="utf-8"))
        people = data.get("people", {})

        return {
            person: [list(map(float, embedding)) for embedding in embeddings]
            for person, embeddings in people.items()
        }

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"people": self.records}, indent=2),
            encoding="utf-8",
        )

    def enroll(self, name: str, embedding: list[float]) -> dict:
        self.records.setdefault(name, []).append(embedding)
        self._save()

        return {
            "success": True,
            "name": name,
            "samples": len(self.records[name]),
        }

    def recognize(self, embedding: list[float]) -> dict:
        best_name = "unknown"
        best_score = 0.0

        for name, embeddings in self.records.items():
            for known_embedding in embeddings:
                score = cosine_similarity(embedding, known_embedding)
                if score > best_score:
                    best_score = score
                    best_name = name

        matched = best_score >= self.threshold

        return {
            "success": True,
            "matched": matched,
            "name": best_name if matched else "unknown",
            "score": best_score,
            "threshold": self.threshold,
            "people_count": len(self.records),
        }

    def list_people(self) -> dict:
        return {
            "success": True,
            "people": [
                {"name": name, "samples": len(embeddings)}
                for name, embeddings in sorted(self.records.items())
            ],
        }
