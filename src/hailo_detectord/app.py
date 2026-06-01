from datetime import datetime, timedelta
import json
from pathlib import Path
import random
from time import perf_counter
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from hailo_detectord.backends import create_backend
from hailo_detectord.config import get_settings
from hailo_detectord.face_recognition import FaceLibrary, compute_embedding
from hailo_detectord.greenhouse import GreenhouseClassifier, WARNING as GREENHOUSE_WARNING
from hailo_detectord.image import decode_image
from hailo_detectord.model_manager import ensure_model
from hailo_detectord.models import ClassifyResponse, DetectResponse, HealthResponse
from hailo_detectord.service import InferenceQueue, metrics, require_api_key
from hailo_detectord.version import version_info


def create_app() -> FastAPI:
    settings = get_settings()

    ensure_model(settings)

    backend = create_backend(settings)
    inference_queue = InferenceQueue(settings.max_concurrent_inferences)
    face_library = FaceLibrary(settings)
    greenhouse_classifier = GreenhouseClassifier(settings)

    app = FastAPI(title="hailo-detectord", version="0.7.0")

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            backend=backend.name,
            model_path=settings.model_path,
            model_metadata_path=settings.model_metadata_path,
        )

    @app.get("/version")
    async def version():
        return version_info(settings, backend)

    @app.get("/metrics")
    async def get_metrics(dep=Depends(require_api_key(settings))):
        snapshot = metrics.snapshot()
        snapshot["model"] = version_info(settings, backend)
        return snapshot

    @app.get("/metrics/prometheus")
    async def get_prometheus_metrics(dep=Depends(require_api_key(settings))):
        return PlainTextResponse(metrics.prometheus(), media_type="text/plain; version=0.0.4")

    def _embedding_from_image_bytes(data: bytes, endpoint: str) -> list[float]:
        try:
            decoded = decode_image(data)
        except ValueError as exc:
            metrics.record_error(endpoint)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return compute_embedding(decoded)

    async def enroll_face_image(data: bytes, name: str, endpoint: str):
        metrics.record_request(endpoint, "face_enroll")
        embedding = _embedding_from_image_bytes(data, endpoint)
        result = face_library.enroll(name=name, embedding=embedding)
        result["embedding_model"] = "test-deterministic-v1"
        result["embedding_dimensions"] = len(embedding)
        result["warning"] = "Development-only deterministic embedding; not production face recognition."
        return result

    async def recognize_face_image(data: bytes, endpoint: str):
        metrics.record_request(endpoint, "face_recognize")
        embedding = _embedding_from_image_bytes(data, endpoint)
        result = face_library.recognize(embedding)
        result["embedding_model"] = "test-deterministic-v1"
        result["embedding_dimensions"] = len(embedding)
        result["warning"] = "Development-only deterministic embedding; not production face recognition."
        return result

    # LAN/internal endpoints intended for Frigate/Double Take style integrations.
    @app.get("/v1/face/library")
    async def list_face_library_internal():
        metrics.record_request("/v1/face/library")
        return face_library.list_people()

    @app.post("/v1/face/enroll")
    async def enroll_face_internal(
        name: str = Form(...),
        image: UploadFile = File(...),
    ):
        return await enroll_face_image(await image.read(), name, "/v1/face/enroll")

    @app.post("/v1/face/recognize")
    async def recognize_face_internal(image: UploadFile = File(...)):
        return await recognize_face_image(await image.read(), "/v1/face/recognize")

    # DeepStack-ish compatibility route for face recognition adapters.
    @app.post("/v1/vision/face/recognize")
    async def recognize_face_deepstack(image: UploadFile = File(...)):
        result = await recognize_face_image(await image.read(), "/v1/vision/face/recognize")
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
        metrics.record_request("/public/v1/face/library")
        return face_library.list_people()

    @app.post("/public/v1/face/enroll")
    async def enroll_face_public(
        name: str = Form(...),
        image: UploadFile = File(...),
        dep=Depends(require_api_key(settings)),
    ):
        return await enroll_face_image(await image.read(), name, "/public/v1/face/enroll")

    @app.post("/public/v1/face/recognize")
    async def recognize_face_public(
        image: UploadFile = File(...),
        dep=Depends(require_api_key(settings)),
    ):
        return await recognize_face_image(await image.read(), "/public/v1/face/recognize")

    @app.post("/public/v1/object/detection", response_model=DetectResponse)
    async def public_object_detect(
        image: UploadFile = File(...),
        dep=Depends(require_api_key(settings)),
    ) -> DetectResponse:
        return await run_detection(
            await image.read(), detector_type="object", endpoint="/public/v1/object/detection"
        )

    @app.post("/public/v1/face/detection", response_model=DetectResponse)
    async def public_face_detect(
        image: UploadFile = File(...),
        dep=Depends(require_api_key(settings)),
    ) -> DetectResponse:
        return await run_detection(
            await image.read(), detector_type="face", endpoint="/public/v1/face/detection"
        )

    @app.post("/public/v1/face/embed")
    async def face_embed(
        image: UploadFile = File(...),
        dep=Depends(require_api_key(settings)),
    ):
        endpoint = "/public/v1/face/embed"
        metrics.record_request(endpoint, "face_embed")
        embedding = _embedding_from_image_bytes(await image.read(), endpoint)
        return {
            "success": True,
            "embedding_model": "test-deterministic-v1",
            "embedding_dimensions": len(embedding),
            "embedding": embedding,
            "warning": "Development-only deterministic embedding; not production face recognition.",
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

    async def run_detection(data: bytes, *, detector_type: str = "object", endpoint: str) -> DetectResponse:
        metrics.record_request(endpoint, detector_type)
        try:
            image = decode_image(data)
        except ValueError as exc:
            metrics.record_error(endpoint)
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        async def infer(queue_wait_ms: float):
            started = perf_counter()

            try:
                if detector_type == "face":
                    predictions = backend.detect_faces(image)
                else:
                    predictions = backend.detect(image)
            except Exception:
                metrics.record_error(endpoint)
                raise

            elapsed_ms = (perf_counter() - started) * 1000

            response = DetectResponse(
                predictions=predictions,
                inference_ms=elapsed_ms,
                backend=f"{backend.name}:{detector_type}",
            )

            metrics.record_inference(detector_type, elapsed_ms, queue_wait_ms)
            save_debug_capture(data, response, detector_type)

            return response

        return await inference_queue.run(infer)

    async def run_greenhouse_classification(data: bytes, endpoint: str) -> ClassifyResponse:
        metrics.record_request(endpoint, "greenhouse_disease")
        try:
            image = decode_image(data)
        except ValueError as exc:
            metrics.record_error(endpoint)
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        started = perf_counter()
        if settings.greenhouse_backend == "hailo":
            if not getattr(backend, "greenhouse", None):
                metrics.record_error(endpoint)
                raise HTTPException(status_code=503, detail="Greenhouse Hailo model not loaded")
            predictions = backend.classify_greenhouse(image)
            backend_name = "hailo:greenhouse"
            warning = None
        else:
            predictions = greenhouse_classifier.classify(image)
            backend_name = greenhouse_classifier.name
            warning = GREENHOUSE_WARNING
        elapsed_ms = (perf_counter() - started) * 1000
        metrics.record_inference("greenhouse_disease", elapsed_ms, 0.0)

        return ClassifyResponse(
            predictions=predictions,
            inference_ms=elapsed_ms,
            backend=backend_name,
            warning=warning,
        )

    @app.post("/detect", response_model=DetectResponse)
    async def detect_raw(request: Request) -> DetectResponse:
        return await run_detection(
            await request.body(), detector_type="object", endpoint="/detect"
        )

    @app.post("/v1/vision/detection", response_model=DetectResponse)
    async def detect_deepstack(image: UploadFile = File(...)) -> DetectResponse:
        return await run_detection(
            await image.read(), detector_type="object", endpoint="/v1/vision/detection"
        )

    @app.post("/v1/object/detection", response_model=DetectResponse)
    async def detect_objects(image: UploadFile = File(...)) -> DetectResponse:
        return await run_detection(
            await image.read(), detector_type="object", endpoint="/v1/object/detection"
        )

    @app.post("/v1/face/detection", response_model=DetectResponse)
    async def detect_faces(image: UploadFile = File(...)) -> DetectResponse:
        return await run_detection(
            await image.read(), detector_type="face", endpoint="/v1/face/detection"
        )

    @app.post("/v1/greenhouse/disease/classify", response_model=ClassifyResponse)
    async def classify_greenhouse_disease(image: UploadFile = File(...)) -> ClassifyResponse:
        return await run_greenhouse_classification(
            await image.read(), endpoint="/v1/greenhouse/disease/classify"
        )

    @app.get("/v1/greenhouse/model/status")
    async def greenhouse_model_status():
        if not hasattr(backend, "greenhouse_status"):
            raise HTTPException(status_code=400, detail="Greenhouse Hailo backend unavailable")
        metrics.record_request("/v1/greenhouse/model/status", "greenhouse_control")
        return backend.greenhouse_status()

    @app.post("/v1/greenhouse/model/load")
    async def greenhouse_model_load():
        if not hasattr(backend, "load_greenhouse"):
            raise HTTPException(status_code=400, detail="Greenhouse Hailo backend unavailable")
        metrics.record_request("/v1/greenhouse/model/load", "greenhouse_control")
        try:
            return backend.load_greenhouse()
        except RuntimeError as exc:
            metrics.record_error("/v1/greenhouse/model/load")
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/v1/greenhouse/model/unload")
    async def greenhouse_model_unload():
        if not hasattr(backend, "unload_greenhouse"):
            raise HTTPException(status_code=400, detail="Greenhouse Hailo backend unavailable")
        metrics.record_request("/v1/greenhouse/model/unload", "greenhouse_control")
        return backend.unload_greenhouse()

    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(request: Request, exc: RuntimeError) -> JSONResponse:
        metrics.record_error(str(request.url.path))
        return JSONResponse(status_code=503, content={"success": False, "error": str(exc)})

    return app


app = create_app()
