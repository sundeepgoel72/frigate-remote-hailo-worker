import asyncio
import os
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest

from hailo_detectord.app import create_app
from hailo_detectord.config import get_settings
from hailo_detectord.service import metrics


class ASGIClient:
    def __init__(self, app) -> None:
        self.app = app

    def get(self, url: str, **kwargs) -> httpx.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs) -> httpx.Response:
        return self.request("POST", url, **kwargs)

    def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        async def run_request() -> httpx.Response:
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                return await client.request(method, url, **kwargs)

        return asyncio.run(run_request())


@pytest.fixture(autouse=True)
def reset_settings_and_metrics(monkeypatch: pytest.MonkeyPatch):
    for key in [name for name in os.environ if name.startswith("HAILO_")]:
        monkeypatch.delenv(key, raising=False)

    get_settings.cache_clear()
    metrics.requests_total = 0
    metrics.detector_requests.clear()
    metrics.endpoint_requests.clear()
    metrics.endpoint_errors.clear()
    metrics.errors_total = 0
    metrics.active_inferences = 0
    metrics.inference_latency_ms.clear()
    metrics.queue_wait_ms.clear()

    yield

    get_settings.cache_clear()
    metrics.requests_total = 0
    metrics.detector_requests.clear()
    metrics.endpoint_requests.clear()
    metrics.endpoint_errors.clear()
    metrics.errors_total = 0
    metrics.active_inferences = 0
    metrics.inference_latency_ms.clear()
    metrics.queue_wait_ms.clear()


def _jpeg(width: int = 320, height: int = 240) -> bytes:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


class _FakeGreenhouseBackend:
    name = "hailo"

    def __init__(self) -> None:
        self.labels = ("person", "car", "dog", "cat")
        self.metadata = {}
        self.hef = None
        self.input_shape = [640, 640, 3]
        self.greenhouse = None

    def greenhouse_status(self) -> dict:
        return {
            "success": True,
            "loaded": self.greenhouse is not None,
            "model_path": "/tmp/greenhouse.hef",
            "backend": "hailo:greenhouse",
        }

    def load_greenhouse(self) -> dict:
        self.greenhouse = object()
        return {
            "success": True,
            "loaded": True,
            "model_path": "/tmp/greenhouse.hef",
            "backend": "hailo:greenhouse",
        }

    def unload_greenhouse(self) -> dict:
        self.greenhouse = None
        return {
            "success": True,
            "loaded": False,
            "model_path": "/tmp/greenhouse.hef",
            "backend": "hailo:greenhouse",
        }

    def classify_greenhouse(self, image: np.ndarray):
        if self.greenhouse is None:
            raise RuntimeError("not loaded")
        from hailo_detectord.models import Classification

        return [
            Classification(label="healthy", confidence=0.9),
            Classification(label="leaf_spot_possible", confidence=0.1),
        ]


def test_health_uses_mock_backend_by_default() -> None:
    client = ASGIClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["backend"] == "mock"


def test_version_endpoint_reports_model_and_face_warning() -> None:
    client = ASGIClient(create_app())

    response = client.get("/version")

    assert response.status_code == 200
    body = response.json()
    assert body["app"] == "hailo-detectord"
    assert body["backend"] == "mock"
    assert body["label_count"] == 4
    assert body["face_backend"] == "dev-deterministic"
    assert "dev-only" in body["face_recognition_warning"]


def test_deepstack_detection_endpoint_returns_predictions() -> None:
    client = ASGIClient(create_app())

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
    client = ASGIClient(create_app())

    response = client.post(
        "/v1/object/detection",
        files={"image": ("frame.jpg", _jpeg(), "image/jpeg")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["backend"] == "mock:object"


def test_face_detection_endpoint_returns_predictions() -> None:
    client = ASGIClient(create_app())

    response = client.post(
        "/v1/face/detection",
        files={"image": ("face.jpg", _jpeg(), "image/jpeg")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["backend"] == "mock:face"


def test_greenhouse_disease_classification_endpoint_returns_ranked_predictions() -> None:
    client = ASGIClient(create_app())

    response = client.post(
        "/v1/greenhouse/disease/classify",
        files={"image": ("leaf.jpg", _jpeg(), "image/jpeg")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["backend"] == "greenhouse-color-baseline"
    assert body["warning"]
    assert [prediction["label"] for prediction in body["predictions"]] == [
        "necrosis_browning",
        "chlorosis_yellowing",
        "leaf_spot_possible",
        "healthy",
    ]


def test_greenhouse_hailo_model_load_classify_and_unload_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HAILO_GREENHOUSE_BACKEND", "hailo")
    monkeypatch.setenv("HAILO_GREENHOUSE_MODEL_PATH", "/tmp/greenhouse.hef")
    monkeypatch.setenv("HAILO_GREENHOUSE_AUTOLOAD", "false")
    monkeypatch.setattr("hailo_detectord.app.create_backend", lambda settings: _FakeGreenhouseBackend())
    get_settings.cache_clear()
    client = ASGIClient(create_app())

    status_before = client.get("/v1/greenhouse/model/status")
    classify_before = client.post(
        "/v1/greenhouse/disease/classify",
        files={"image": ("leaf.jpg", _jpeg(), "image/jpeg")},
    )
    load = client.post("/v1/greenhouse/model/load")
    classify_after = client.post(
        "/v1/greenhouse/disease/classify",
        files={"image": ("leaf.jpg", _jpeg(), "image/jpeg")},
    )
    unload = client.post("/v1/greenhouse/model/unload")
    status_after = client.get("/v1/greenhouse/model/status")

    assert status_before.json()["loaded"] is False
    assert classify_before.status_code == 503
    assert load.json()["loaded"] is True
    assert classify_after.status_code == 200
    assert classify_after.json()["backend"] == "hailo:greenhouse"
    assert unload.json()["loaded"] is False
    assert status_after.json()["loaded"] is False


def test_raw_detection_endpoint_rejects_invalid_image() -> None:
    client = ASGIClient(create_app())

    response = client.post("/detect", content=b"not an image")

    assert response.status_code == 400


def test_metrics_are_protected_when_public_api_disabled() -> None:
    client = ASGIClient(create_app())

    response = client.get("/metrics")

    assert response.status_code == 403
    assert response.json()["detail"] == "Public API disabled"


def test_public_object_endpoint_accepts_direct_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HAILO_PUBLIC_API_ENABLED", "true")
    monkeypatch.setenv("HAILO_API_KEYS", '["test-secret"]')
    get_settings.cache_clear()
    client = ASGIClient(create_app())

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
    client = ASGIClient(create_app())

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
    client = ASGIClient(create_app())

    client.post("/v1/object/detection", files={"image": ("frame.jpg", _jpeg(), "image/jpeg")})
    client.post("/v1/face/detection", files={"image": ("face.jpg", _jpeg(), "image/jpeg")})

    response = client.get("/metrics", headers={"X-API-Key": "metrics-secret"})

    assert response.status_code == 200
    body = response.json()
    assert body["requests_total"] == 2
    assert body["detector_requests"] == {"object": 1, "face": 1}
    assert body["endpoint_requests"] == {
        "/v1/object/detection": 1,
        "/v1/face/detection": 1,
    }
    assert body["inference_latency_ms"]["count"] == 2
    assert body["queue_wait_ms"]["count"] == 2
    assert body["errors_total"] == 0
    assert body["model"]["backend"] == "mock"


def test_prometheus_metrics_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HAILO_PUBLIC_API_ENABLED", "true")
    monkeypatch.setenv("HAILO_API_KEYS", '["prom-secret"]')
    get_settings.cache_clear()
    client = ASGIClient(create_app())

    client.post("/v1/object/detection", files={"image": ("frame.jpg", _jpeg(), "image/jpeg")})

    response = client.get("/metrics/prometheus", headers={"X-API-Key": "prom-secret"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "hailo_requests_total 1" in response.text
    assert 'hailo_detector_requests_total{detector="object"} 1' in response.text


def test_face_enroll_recognize_and_deepstack_compatibility(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HAILO_FACE_LIBRARY_PATH", str(tmp_path / "faces.json"))
    get_settings.cache_clear()
    client = ASGIClient(create_app())

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
    client = ASGIClient(create_app())

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
    client = ASGIClient(create_app())

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
