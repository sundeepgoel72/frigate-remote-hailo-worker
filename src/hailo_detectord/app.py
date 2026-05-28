from datetime import datetime, timedelta
import json
from pathlib import Path
import random
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

    app = FastAPI(title="hailo-detectord", version="0.4.0")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            backend=backend.name,
            model_path=settings.model_path,
            model_metadata_path=settings.model_metadata_path,
        )

    def cleanup_debug_captures(base_dir: Path) -> None:
        if not base_dir.exists():
            return

        files = sorted(
            [path for path in base_dir.rglob("*") if path.is_file()],
            key=lambda path: path.stat().st_mtime,
        )

        if settings.debug_capture_max_age_days > 0:
            cutoff = datetime.utcnow() - timedelta(days=settings.debug_capture_max_age_days)
            for file_path in files:
                modified = datetime.utcfromtimestamp(file_path.stat().st_mtime)
                if modified < cutoff:
                    file_path.unlink(missing_ok=True)

        if settings.debug_capture_max_bytes > 0:
            files = sorted(
                [path for path in base_dir.rglob("*") if path.is_file()],
                key=lambda path: path.stat().st_mtime,
            )

            total_size = sum(path.stat().st_size for path in files)

            while total_size > settings.debug_capture_max_bytes and files:
                oldest = files.pop(0)
                size = oldest.stat().st_size
                oldest.unlink(missing_ok=True)
                total_size -= size

    def should_capture(response: DetectResponse) -> bool:
        if not settings.debug_capture_enabled:
            return False

        if settings.debug_capture_sample_rate < 1.0:
            if random.random() > settings.debug_capture_sample_rate:
                return False

        if settings.debug_capture_failed_only and response.predictions:
            return False

        if settings.debug_capture_labels:
            prediction_labels = {prediction.label for prediction in response.predictions}
            configured_labels = set(settings.debug_capture_labels)

            if not prediction_labels.intersection(configured_labels):
                return False

        return True

    def save_debug_capture(data: bytes, response: DetectResponse, detector_type: str) -> None:
        if not should_capture(response):
            return

        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")
        capture_id = uuid4().hex[:8]

        capture_dir = Path(settings.debug_capture_dir) / detector_type
        capture_dir.mkdir(parents=True, exist_ok=True)

        cleanup_debug_captures(Path(settings.debug_capture_dir))

        image_path = capture_dir / f"{timestamp}-{capture_id}.jpg"
        metadata_path = capture_dir / f"{timestamp}-{capture_id}.json"

        image_path.write_bytes(data)

        metadata = {
            "timestamp": timestamp,
            "capture_id": capture_id,
            "detector_type": detector_type,
            "backend": response.backend,
            "inference_ms": response.inference_ms,
            "prediction_count": len(response.predictions),
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
