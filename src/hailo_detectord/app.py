from datetime import datetime, timedelta
import json
from pathlib import Path
import random
from time import perf_counter
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from hailo_detectord.backends import create_backend
from hailo_detectord.config import get_settings
from hailo_detectord.face_recognition import FaceLibrary, compute_embedding
from hailo_detectord.image import decode_image
from hailo_detectord.model_manager import ensure_model
from hailo_detectord.models import DetectResponse, HealthResponse
from hailo_detectord.service import InferenceQueue, metrics, require_api_key


def create_app() -> FastAPI:
    settings = get_settings()

    ensure_model(settings)

    backend = create_backend(settings)
    inference_queue = InferenceQueue(settings.max_concurrent_inferences)
    face_library = FaceLibrary(settings)

    app = FastAPI(title="hailo-detectord", version="0.6.0")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            backend=backend.name,
            model_path=settings.model_path,
            model_metadata_path=settings.model_metadata_path,
        )

    @app.get("/metrics")
    async def get_metrics(dep=Depends(require_api_key(settings))):
        return {
            "requests_total": metrics.requests_total,
            "detector_requests": dict(metrics.detector_requests),
            "errors_total": metrics.errors_total,
        }

    def _embedding_from_image_bytes(data: bytes) -> list[float]:
        try:
            decoded = decode_image(data)
        except ValueError as exc:
            metrics.errors_total += 1
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return compute_embedding(decoded)

    async def enroll_face_image(data: bytes, name: str):
        embedding = _embedding_from_image_bytes(data)
        result = face_library.enroll(name=name, embedding=embedding)
        result["embedding_model"] = "test-deterministic-v1"
        result["embedding_dimensions"] = len(embedding)
        return result

    async def recognize_face_image(data: bytes):
        embedding = _embedding_from_image_bytes(data)
        result = face_library.recognize(embedding)
        result["embedding_model"] = "test-deterministic-v1"
        result["embedding_dimensions"] = len(embedding)
        return result

    # LAN/internal endpoints intended for Frigate/Double Take style integrations.
    @app.get("/v1/face/library")
    async def list_face_library_internal():
        return face_library.list_people()

    @app.post("/v1/face/enroll")
    async def enroll_face_internal(
        name: str = Form(...),
        image: UploadFile = File(...),
    ):
        return await enroll_face_image(await image.read(), name)

    @app.post("/v1/face/recognize")
    async def recognize_face_internal(image: UploadFile = File(...)):
        return await recognize_face_image(await image.read())

    # DeepStack-ish compatibility route for face recognition adapters.
    @app.post("/v1/vision/face/recognize")
    async def recognize_face_deepstack(image: UploadFile = File(...)):
        result = await recognize_face_image(await image.read())
        return {
            "success": result["success"],
            "predictions": [
                {
                    "userid": result["name"],
                    "confidence": result["score"],
                    "matched": result["matched"],
                }
            ],
        }

    # Protected/public endpoints intended for external exposure.
    @app.get("/public/v1/face/library")
    async def list_face_library_public(dep=Depends(require_api_key(settings))):
        return face_library.list_people()

    @app.post("/public/v1/face/enroll")
    async def enroll_face_public(
        name: str = Form(...),
        image: UploadFile = File(...),
        dep=Depends(require_api_key(settings)),
    ):
        return await enroll_face_image(await image.read(), name)

    @app.post("/public/v1/face/recognize")
    async def recognize_face_public(
        image: UploadFile = File(...),
        dep=Depends(require_api_key(settings)),
    ):
        return await recognize_face_image(await image.read())

    @app.post("/public/v1/object/detection", response_model=DetectResponse)
    async def public_object_detect(
        image: UploadFile = File(...),
        dep=Depends(require_api_key(settings)),
    ) -> DetectResponse:
        return await run_detection(await image.read(), detector_type="object")

    @app.post("/public/v1/face/detection", response_model=DetectResponse)
    async def public_face_detect(
        image: UploadFile = File(...),
        dep=Depends(require_api_key(settings)),
    ) -> DetectResponse:
        return await run_detection(await image.read(), detector_type="face")

    @app.post("/public/v1/face/embed")
    async def face_embed(
        image: UploadFile = File(...),
        dep=Depends(require_api_key(settings)),
    ):
        embedding = _embedding_from_image_bytes(await image.read())
        return {
            "success": True,
            "embedding_model": "test-deterministic-v1",
            "embedding_dimensions": len(embedding),
            "embedding": embedding,
        }

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
            metrics.errors_total += 1
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        async def infer():
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

            metrics.record(detector_type)
            save_debug_capture(data, response, detector_type)

            return response

        return await inference_queue.run(infer)

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
        metrics.errors_total += 1
        return JSONResponse(status_code=503, content={"success": False, "error": str(exc)})

    return app


app = create_app()
