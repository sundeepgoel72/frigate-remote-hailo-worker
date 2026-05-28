from datetime import datetime
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from hailo_detectord.backends import create_backend
from hailo_detectord.config import get_settings
from hailo_detectord.image import decode_image
from hailo_detectord.models import DetectResponse, HealthResponse


def create_app() -> FastAPI:
    settings = get_settings()
    backend = create_backend(settings)

    app = FastAPI(title="hailo-detectord", version="0.3.0")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            backend=backend.name,
            model_path=settings.model_path,
            model_metadata_path=settings.model_metadata_path,
        )

    def save_debug_capture(data: bytes, response: DetectResponse, detector_type: str) -> None:
        if not settings.debug_capture_enabled:
            return

        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")
        capture_id = uuid4().hex[:8]

        capture_dir = Path(settings.debug_capture_dir) / detector_type
        capture_dir.mkdir(parents=True, exist_ok=True)

        image_path = capture_dir / f"{timestamp}-{capture_id}.jpg"
        metadata_path = capture_dir / f"{timestamp}-{capture_id}.json"

        image_path.write_bytes(data)

        metadata = {
            "detector_type": detector_type,
            "backend": response.backend,
            "inference_ms": response.inference_ms,
            "predictions": [prediction.model_dump() for prediction in response.predictions],
        }

        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    async def run_detection(data: bytes, *, detector_type: str = "object") -> DetectResponse:
        try:
            image = decode_image(data)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        started = perf_counter()

        if detector_type == "face":
            predictions = backend.detect_faces(image)
        else:
            predictions = backend.detect(image)

        elapsed_ms = (perf_counter() - started) * 1000

        response = DetectResponse(
            predictions=predictions,
            inference_ms=elapsed_ms,
            backend=f"{backend.name}:{detector_type}",
        )

        save_debug_capture(data, response, detector_type)

        return response

    @app.post("/detect", response_model=DetectResponse)
    async def detect_raw(request: Request) -> DetectResponse:
        return await run_detection(await request.body(), detector_type="object")

    @app.post("/v1/vision/detection", response_model=DetectResponse)
    async def detect_deepstack(image: UploadFile = File(...)) -> DetectResponse:
        return await run_detection(await image.read(), detector_type="object")

    @app.post("/v1/object/detection", response_model=DetectResponse)
    async def detect_objects(image: UploadFile = File(...)) -> DetectResponse:
        return await run_detection(await image.read(), detector_type="object")

    @app.post("/v1/face/detection", response_model=DetectResponse)
    async def detect_faces(image: UploadFile = File(...)) -> DetectResponse:
        return await run_detection(await image.read(), detector_type="face")

    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(_request: Request, exc: RuntimeError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"success": False, "error": str(exc)})

    return app


app = create_app()
