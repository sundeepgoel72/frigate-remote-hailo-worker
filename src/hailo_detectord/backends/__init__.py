from hailo_detectord.backends.base import DetectorBackend
from hailo_detectord.backends.hailo import HailoBackend
from hailo_detectord.backends.mock import MockBackend
from hailo_detectord.config import Settings


def create_backend(settings: Settings) -> DetectorBackend:
    if settings.backend == "hailo":
        return HailoBackend(settings)
    return MockBackend(settings)


__all__ = ["DetectorBackend", "create_backend"]
