import asyncio
from collections import Counter
from typing import Callable

from fastapi import Header, HTTPException

from hailo_detectord.config import Settings


class Metrics:
    def __init__(self) -> None:
        self.requests_total = 0
        self.detector_requests = Counter()
        self.errors_total = 0

    def record(self, detector_type: str) -> None:
        self.requests_total += 1
        self.detector_requests[detector_type] += 1


metrics = Metrics()


class InferenceQueue:
    def __init__(self, concurrency: int) -> None:
        self.semaphore = asyncio.Semaphore(concurrency)

    async def run(self, fn: Callable):
        async with self.semaphore:
            return await fn()


def require_api_key(settings: Settings):
    async def dependency(
        x_api_key: str | None = Header(default=None),
        x_rapidapi_proxy_secret: str | None = Header(default=None),
        x_rapidapi_user: str | None = Header(default=None),
        x_rapidapi_subscription: str | None = Header(default=None),
    ):
        if not settings.public_api_enabled:
            raise HTTPException(status_code=403, detail="Public API disabled")

        # RapidAPI provider mode.
        if settings.rapidapi_enabled:
            if not settings.rapidapi_proxy_secret:
                raise HTTPException(status_code=503, detail="RapidAPI proxy secret not configured")

            if x_rapidapi_proxy_secret == settings.rapidapi_proxy_secret:
                return {
                    "auth_type": "rapidapi",
                    "rapidapi_user": x_rapidapi_user,
                    "rapidapi_subscription": x_rapidapi_subscription,
                }

        # Direct API-key mode.
        if settings.api_keys:
            if x_api_key in settings.api_keys:
                return {
                    "auth_type": "direct",
                    "api_key": x_api_key,
                }

        raise HTTPException(status_code=401, detail="Invalid API credentials")

    return dependency
