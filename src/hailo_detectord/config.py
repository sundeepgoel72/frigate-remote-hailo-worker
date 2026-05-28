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

    face_model_path: str | None = None
    face_model_metadata_path: str | None = None
    face_library_path: str | None = None
    face_match_threshold: float = Field(default=0.65, ge=0.0, le=1.0)

    model_download_enabled: bool = False
    model_download_url: str | None = None
    model_download_path: str | None = None
    model_download_token: str | None = None
    model_download_force: bool = False

    api_keys: tuple[str, ...] = ()
    public_api_enabled: bool = False

    rapidapi_enabled: bool = False
    rapidapi_proxy_secret: str | None = None

    max_concurrent_inferences: int = Field(default=1, ge=1)

    mqtt_enabled: bool = False
    mqtt_host: str | None = None
    mqtt_port: int = Field(default=1883, ge=1, le=65535)
    mqtt_username: str | None = None
    mqtt_password: str | None = None
    mqtt_topic_prefix: str = "hailo-detectord"

    debug_capture_enabled: bool = False
    debug_capture_dir: str = "/tmp/hailo-detectord-captures"
    debug_capture_sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    debug_capture_failed_only: bool = False
    debug_capture_labels: tuple[str, ...] = ()
    debug_capture_max_age_days: int = Field(default=0, ge=0)
    debug_capture_max_bytes: int = Field(default=0, ge=0)

    def metadata(self) -> dict:
        if not self.model_metadata_path:
            return {}
        return json.loads(Path(self.model_metadata_path).read_text(encoding="utf-8"))


@lru_cache
def get_settings() -> Settings:
    return Settings()
