from importlib import metadata, util
import os
from pathlib import Path
import subprocess
from typing import Any

from hailo_detectord.config import Settings

APP_VERSION = "0.7.0"


def _git_commit() -> str | None:
    if value := os.environ.get("HAILO_GIT_COMMIT"):
        return value

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except Exception:
        return None


def _package_version() -> str:
    try:
        return metadata.version("frigate-remote-hailo-worker")
    except metadata.PackageNotFoundError:
        return APP_VERSION


def _model_id(settings: Settings, backend: Any) -> str | None:
    backend_metadata = getattr(backend, "metadata", {}) or {}

    for key in ("modelId", "model_id", "name", "modelName"):
        if value := backend_metadata.get(key):
            return str(value)

    if settings.model_path:
        return Path(settings.model_path).name

    return None


def _output_shapes(backend: Any) -> list[dict[str, Any]]:
    hef = getattr(backend, "hef", None)
    if hef is None:
        return []

    try:
        outputs = []
        for info in hef.get_output_vstream_infos():
            outputs.append(
                {
                    "name": info.name,
                    "shape": list(info.shape),
                    "format": str(info.format.type),
                }
            )
        return outputs
    except Exception:
        return []


def version_info(settings: Settings, backend: Any) -> dict[str, Any]:
    labels = getattr(backend, "labels", None) or list(settings.labels)

    return {
        "app": "hailo-detectord",
        "app_version": _package_version(),
        "git_commit": _git_commit(),
        "backend": getattr(backend, "name", "unknown"),
        "model_path": settings.model_path,
        "model_metadata_path": settings.model_metadata_path,
        "model_id": _model_id(settings, backend),
        "input_shape": list(getattr(backend, "input_shape", []) or []),
        "output_shapes": _output_shapes(backend),
        "label_count": len(labels),
        "hailort_available": util.find_spec("hailo_platform") is not None,
        "greenhouse_backend": settings.greenhouse_backend,
        "greenhouse_model_path": settings.greenhouse_model_path,
        "greenhouse_model_loaded": getattr(backend, "greenhouse", None) is not None,
        "face_backend": "dev-deterministic" if not settings.face_model_path else "configured",
        "face_model_path": settings.face_model_path,
        "face_recognition_warning": (
            "Current deterministic face embedding is dev-only; replace with ArcFace, "
            "InsightFace, or a Hailo embedding model before production use."
        ),
    }
