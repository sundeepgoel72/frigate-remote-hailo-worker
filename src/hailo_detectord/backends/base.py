from abc import ABC, abstractmethod

import numpy as np

from hailo_detectord.models import Detection


class DetectorBackend(ABC):
    name: str

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[Detection]:
        """Run object inference on an OpenCV BGR image."""

    def detect_faces(self, image: np.ndarray) -> list[Detection]:
        """Run face inference on an OpenCV BGR image.

        Backends can override this when a dedicated face model is configured.
        The default keeps the API usable with a single loaded detector while the
        service contract is developed.
        """
        return self.detect(image)
