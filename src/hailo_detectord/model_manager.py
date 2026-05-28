from pathlib import Path
from urllib.request import Request, urlopen

from hailo_detectord.config import Settings


def ensure_model(settings: Settings) -> None:
    if not settings.model_download_enabled:
        return

    if not settings.model_download_url or not settings.model_download_path:
        return

    model_path = Path(settings.model_download_path)

    if model_path.exists() and not settings.model_download_force:
        return

    model_path.parent.mkdir(parents=True, exist_ok=True)

    headers = {}
    if settings.model_download_token:
        headers["Authorization"] = f"Bearer {settings.model_download_token}"

    request = Request(settings.model_download_url, headers=headers)

    with urlopen(request, timeout=60) as response:
        model_path.write_bytes(response.read())
