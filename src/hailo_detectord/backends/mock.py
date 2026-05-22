import numpy as np

from hailo_detectord.backends.base import DetectorBackend
from hailo_detectord.config import Settings
from hailo_detectord.models import Detection


class MockBackend(DetectorBackend):
    name = "mock"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def detect(self, image: np.ndarray) -> list[Detection]:
        height, width = image.shape[:2]
        label = self.settings.labels[0] if self.settings.labels else "object"
        return [
            Detection(
                label=label,
                confidence=max(self.settings.confidence_threshold, 0.80),
                x_min=width // 4,
                y_min=height // 5,
                x_max=(width * 3) // 4,
                y_max=(height * 4) // 5,
            )
        ]
