import cv2
import numpy as np
from fastapi.testclient import TestClient

from hailo_detectord.app import create_app


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
