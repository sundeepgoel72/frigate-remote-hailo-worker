import os
from pathlib import Path

import httpx
import pytest


pytestmark = pytest.mark.skipif(
    os.getenv("HAILO_INTEGRATION") != "1",
    reason="set HAILO_INTEGRATION=1 to run live hailo-detectord integration tests",
)


def test_live_hailo_service_metadata() -> None:
    base_url = os.getenv("HAILO_INTEGRATION_BASE_URL", "http://127.0.0.1:32168")

    with httpx.Client(base_url=base_url, timeout=5.0) as client:
        health = client.get("/health")
        version = client.get("/version")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["backend"] == "hailo"

    assert version.status_code == 200
    body = version.json()
    assert body["backend"] == "hailo"
    assert body["hailort_available"] is True
    assert body["label_count"] > 0


def test_live_hailo_detection_crop() -> None:
    image_path = os.getenv("HAILO_INTEGRATION_IMAGE")
    if not image_path:
        pytest.skip("set HAILO_INTEGRATION_IMAGE to run live detection against a crop")

    path = Path(image_path)
    if not path.is_file():
        pytest.fail(f"HAILO_INTEGRATION_IMAGE does not exist: {path}")

    base_url = os.getenv("HAILO_INTEGRATION_BASE_URL", "http://127.0.0.1:32168")

    with httpx.Client(base_url=base_url, timeout=10.0) as client:
        with path.open("rb") as image:
            response = client.post(
                "/v1/vision/detection",
                files={"image": (path.name, image, "image/jpeg")},
            )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["backend"] == "hailo:object"
    assert isinstance(body["predictions"], list)
