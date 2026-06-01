import cv2
import numpy as np

from hailo_detectord.config import Settings
from hailo_detectord.models import Classification


WARNING = (
    "Baseline color-health classifier for greenhouse API validation; "
    "not a production disease diagnosis model."
)


class GreenhouseClassifier:
    name = "greenhouse-color-baseline"

    def __init__(self, settings: Settings) -> None:
        self.labels = settings.greenhouse_labels

    def classify(self, image: np.ndarray) -> list[Classification]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        green = self._ratio(hsv, (35, 35, 35), (90, 255, 255))
        yellow = self._ratio(hsv, (18, 45, 45), (34, 255, 255))
        brown = self._ratio(hsv, (5, 35, 25), (24, 255, 180))
        dark = float(np.mean(hsv[:, :, 2] < 45))

        scores = {
            "healthy": self._clamp(0.2 + green - max(yellow, brown, dark) * 0.45),
            "chlorosis_yellowing": self._clamp(0.2 + yellow * 1.6),
            "necrosis_browning": self._clamp(0.15 + max(brown, dark * 0.6) * 1.7),
            "leaf_spot_possible": self._clamp(0.1 + min(brown + yellow, 0.7)),
        }

        predictions = [
            Classification(label=label, confidence=scores.get(label, 0.0))
            for label in self.labels
        ]
        return sorted(predictions, key=lambda prediction: prediction.confidence, reverse=True)

    def _ratio(
        self,
        hsv: np.ndarray,
        lower: tuple[int, int, int],
        upper: tuple[int, int, int],
    ) -> float:
        mask = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
        return float(np.mean(mask > 0))

    def _clamp(self, value: float) -> float:
        return max(0.0, min(1.0, value))
