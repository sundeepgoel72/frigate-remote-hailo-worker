import os
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from hailo_detectord.app import create_app
from hailo_detectord.config import get_settings
from hailo_detectord.service import metrics


@pytest.fixture(autouse=True)
def reset_settings_and_metrics(monkeypatch: pytest.MonkeyPatch):
    for key in [name for name in os.environ if name.startswith("HAILO_")]:
        monkeypatch.delenv(key, raising=False)

    get_settings.cache_clear()
    metrics.requests_total = 0
    metrics.detector_requests.clear()
    metrics.errors_total = 0

    yield

    get_settings.cache_clear()
    metrics.requests_total = 0
    metrics.detector_requests.clear()
    metrics.errors_total = 0


def _jpeg(width: int = 320, height: int = 240) -> bytes:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def test_health_uses_mock_backend_by_default() -> None:
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["backend"] == "mock"


def test_deepstack_detection_endpoint_returns_predictions() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/v1/vision/detection",
        files={"image": ("frame.jpg", _jpeg(), "image/jpeg")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["backend"] == "mock:object"
    assert body["predictions"][0]["label"] == "person"


def test_object_detection_alias_returns_predictions() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/v1/object/detection",
        files={"image": ("frame.jpg", _jpeg(), "image/jpeg")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["backend"] == "mock:object"


def test_face_detection_endpoint_returns_predictions() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/v1/face/detection",
        files={"image": ("face.jpg", _jpeg(), "image/jpeg")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["backend"] == "mock:face"


def test_raw_detection_endpoint_rejects_invalid_image() -> None:
    client = TestClient(create_app())

    response = client.post("/detect", content=b"not an image")

    assert response.status_code == 400


def test_metrics_are_protected_when_public_api_disabled() -> None:
    client = TestClient(create_app())

    response = client.get("/metrics")

    assert response.status_code == 403
    assert response.json()["detail"] == "Public API disabled"


def test_public_object_endpoint_accepts_direct_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HAILO_PUBLIC_API_ENABLED", "true")
    monkeypatch.setenv("HAILO_API_KEYS", '["test-secret"]')
    get_settings.cache_clear()
    client = TestClient(create_app())

    rejected = client.post(
        "/public/v1/object/detection",
        files={"image": ("frame.jpg", _jpeg(), "image/jpeg")},
    )
    accepted = client.post(
        "/public/v1/object/detection",
        headers={"X-API-Key": "test-secret"},
        files={"image": ("frame.jpg", _jpeg(), "image/jpeg")},
    )

    assert rejected.status_code == 401
    assert accepted.status_code == 200
    assert accepted.json()["backend"] == "mock:object"


def test_public_endpoint_accepts_rapidapi_proxy_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HAILO_PUBLIC_API_ENABLED", "true")
    monkeypatch.setenv("HAILO_RAPIDAPI_ENABLED", "true")
    monkeypatch.setenv("HAILO_RAPIDAPI_PROXY_SECRET", "rapid-secret")
    get_settings.cache_clear()
    client = TestClient(create_app())

    response = client.post(
        "/public/v1/face/detection",
        headers={"X-RapidAPI-Proxy-Secret": "rapid-secret"},
        files={"image": ("face.jpg", _jpeg(), "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json()["backend"] == "mock:face"


def test_metrics_count_object_and_face_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HAILO_PUBLIC_API_ENABLED", "true")
    monkeypatch.setenv("HAILO_API_KEYS", '["metrics-secret"]')
    get_settings.cache_clear()
    client = TestClient(create_app())

    client.post("/v1/object/detection", files={"image": ("frame.jpg", _jpeg(), "image/jpeg")})
    client.post("/v1/face/detection", files={"image": ("face.jpg", _jpeg(), "image/jpeg")})

    response = client.get("/metrics", headers={"X-API-Key": "metrics-secret"})

    assert response.status_code == 200
    body = response.json()
    assert body["requests_total"] == 2
    assert body["detector_requests"] == {"object": 1, "face": 1}
    assert body["errors_total"] == 0


def test_face_enroll_recognize_and_deepstack_compatibility(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HAILO_FACE_LIBRARY_PATH", str(tmp_path / "faces.json"))
    get_settings.cache_clear()
    client = TestClient(create_app())

    enroll = client.post(
        "/v1/face/enroll",
        data={"name": "Sundeep"},
        files={"image": ("face.jpg", _jpeg(), "image/jpeg")},
    )
    library = client.get("/v1/face/library")
    recognize = client.post(
        "/v1/face/recognize",
        files={"image": ("face.jpg", _jpeg(), "image/jpeg")},
    )
    deepstack = client.post(
        "/v1/vision/face/recognize",
        files={"image": ("face.jpg", _jpeg(), "image/jpeg")},
    )

    assert enroll.status_code == 200
    assert enroll.json()["samples"] == 1
    assert library.json()["people"] == [{"name": "Sundeep", "samples": 1}]
    assert recognize.json()["matched"] is True
    assert recognize.json()["name"] == "Sundeep"
    assert deepstack.json()["predictions"][0]["userid"] == "Sundeep"


def test_public_face_embed_returns_deterministic_embedding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HAILO_PUBLIC_API_ENABLED", "true")
    monkeypatch.setenv("HAILO_API_KEYS", '["embed-secret"]')
    get_settings.cache_clear()
    client = TestClient(create_app())

    first = client.post(
        "/public/v1/face/embed",
        headers={"X-API-Key": "embed-secret"},
        files={"image": ("face.jpg", _jpeg(), "image/jpeg")},
    )
    second = client.post(
        "/public/v1/face/embed",
        headers={"X-API-Key": "embed-secret"},
        files={"image": ("face.jpg", _jpeg(), "image/jpeg")},
    )

    assert first.status_code == 200
    assert first.json()["embedding_dimensions"] > 0
    assert first.json()["embedding"] == second.json()["embedding"]


def test_debug_capture_writes_image_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    capture_dir = tmp_path / "captures"
    monkeypatch.setenv("HAILO_DEBUG_CAPTURE_ENABLED", "true")
    monkeypatch.setenv("HAILO_DEBUG_CAPTURE_DIR", str(capture_dir))
    monkeypatch.setenv("HAILO_DEBUG_CAPTURE_SAMPLE_RATE", "1.0")
    get_settings.cache_clear()
    client = TestClient(create_app())

    response = client.post(
        "/v1/object/detection",
        files={"image": ("frame.jpg", _jpeg(), "image/jpeg")},
    )

    assert response.status_code == 200
    object_dir = capture_dir / "object"
    images = list(object_dir.glob("*.jpg"))
    metadata_files = list(object_dir.glob("*.json"))
    assert len(images) == 1
    assert len(metadata_files) == 1
    assert '"detector_type": "object"' in metadata_files[0].read_text(encoding="utf-8")
