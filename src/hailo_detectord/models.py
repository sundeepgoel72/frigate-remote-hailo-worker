from pydantic import BaseModel, Field


class Detection(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    x_min: int = Field(ge=0)
    y_min: int = Field(ge=0)
    x_max: int = Field(ge=0)
    y_max: int = Field(ge=0)


class DetectResponse(BaseModel):
    success: bool = True
    predictions: list[Detection]
    inference_ms: float
    backend: str


class HealthResponse(BaseModel):
    status: str
    backend: str
    model_path: str | None = None
    model_metadata_path: str | None = None
