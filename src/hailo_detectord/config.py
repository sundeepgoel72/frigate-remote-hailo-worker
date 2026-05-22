from functools import lru_cache
import json
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HAILO_", env_file=".env", extra="ignore")

    backend: Literal["mock", "hailo"] = "mock"
    model_path: str | None = None
    model_metadata_path: str | None = None
    labelmap_path: str | None = None
    bbox_order: Literal["yxyx", "xyxy"] = "yxyx"
    input_pixel_format: Literal["bgr", "rgb"] = "rgb"
    confidence_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    labels: tuple[str, ...] = ("person", "car", "dog", "cat")

    def metadata(self) -> dict:
        if not self.model_metadata_path:
            return {}
        return json.loads(Path(self.model_metadata_path).read_text(encoding="utf-8"))


@lru_cache
def get_settings() -> Settings:
    return Settings()
