import asyncio
from collections import Counter
from functools import wraps
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
    async def dependency(x_api_key: str | None = Header(default=None)):
        if not settings.public_api_enabled:
            raise HTTPException(status_code=403, detail="Public API disabled")

        if not settings.api_keys:
            raise HTTPException(status_code=503, detail="No API keys configured")

        if x_api_key not in settings.api_keys:
            raise HTTPException(status_code=401, detail="Invalid API key")

    return dependency
