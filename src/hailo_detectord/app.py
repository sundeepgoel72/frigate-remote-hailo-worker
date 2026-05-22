from time import perf_counter

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from hailo_detectord.backends import create_backend
from hailo_detectord.config import get_settings
from hailo_detectord.image import decode_image
from hailo_detectord.models import DetectResponse, HealthResponse


def create_app() -> FastAPI:
    settings = get_settings()
    backend = create_backend(settings)

    app = FastAPI(title="hailo-detectord", version="0.1.0")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            backend=backend.name,
            model_path=settings.model_path,
            model_metadata_path=settings.model_metadata_path,
        )

    async def run_detection(data: bytes) -> DetectResponse:
        try:
            image = decode_image(data)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        started = perf_counter()
        predictions = backend.detect(image)
        elapsed_ms = (perf_counter() - started) * 1000
        return DetectResponse(
            predictions=predictions,
            inference_ms=elapsed_ms,
            backend=backend.name,
        )

    @app.post("/detect", response_model=DetectResponse)
    async def detect_raw(request: Request) -> DetectResponse:
        return await run_detection(await request.body())

    @app.post("/v1/vision/detection", response_model=DetectResponse)
    async def detect_deepstack(image: UploadFile = File(...)) -> DetectResponse:
        return await run_detection(await image.read())

    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(_request: Request, exc: RuntimeError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"success": False, "error": str(exc)})

    return app


app = create_app()
