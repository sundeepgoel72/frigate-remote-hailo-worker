from abc import ABC, abstractmethod

import numpy as np

from hailo_detectord.models import Detection


class DetectorBackend(ABC):
    name: str

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[Detection]:
        """Run inference on an OpenCV BGR image."""
