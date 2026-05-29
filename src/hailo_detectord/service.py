import asyncio
from collections import Counter, deque
from time import perf_counter
from typing import Callable

from fastapi import Header, HTTPException

from hailo_detectord.config import Settings


class Metrics:
    def __init__(self) -> None:
        self.requests_total = 0
        self.detector_requests = Counter()
        self.endpoint_requests = Counter()
        self.endpoint_errors = Counter()
        self.errors_total = 0
        self.active_inferences = 0
        self.inference_latency_ms = deque(maxlen=5000)
        self.queue_wait_ms = deque(maxlen=5000)

    def record_request(self, endpoint: str, detector_type: str | None = None) -> None:
        self.requests_total += 1
        self.endpoint_requests[endpoint] += 1

    def record_error(self, endpoint: str) -> None:
        self.errors_total += 1
        self.endpoint_errors[endpoint] += 1

    def record_inference(self, detector_type: str, latency_ms: float, queue_wait_ms: float) -> None:
        self.detector_requests[detector_type] += 1
        self.inference_latency_ms.append(float(latency_ms))
        self.queue_wait_ms.append(float(queue_wait_ms))

    def _percentile(self, values: list[float], percentile: float) -> float:
        if not values:
            return 0.0
        values = sorted(values)
        index = min(len(values) - 1, round((percentile / 100.0) * (len(values) - 1)))
        return float(values[index])

    def snapshot(self) -> dict:
        inference_values = list(self.inference_latency_ms)
        queue_values = list(self.queue_wait_ms)
        return {
            "requests_total": self.requests_total,
            "detector_requests": dict(self.detector_requests),
            "endpoint_requests": dict(self.endpoint_requests),
            "endpoint_errors": dict(self.endpoint_errors),
            "errors_total": self.errors_total,
            "active_inferences": self.active_inferences,
            "inference_latency_ms": {
                "count": len(inference_values),
                "p50": self._percentile(inference_values, 50),
                "p95": self._percentile(inference_values, 95),
                "p99": self._percentile(inference_values, 99),
            },
            "queue_wait_ms": {
                "count": len(queue_values),
                "p50": self._percentile(queue_values, 50),
                "p95": self._percentile(queue_values, 95),
                "p99": self._percentile(queue_values, 99),
            },
        }

    def prometheus(self) -> str:
        snapshot = self.snapshot()
        lines = [
            "# HELP hailo_requests_total Total API requests.",
            "# TYPE hailo_requests_total counter",
            f"hailo_requests_total {snapshot['requests_total']}",
            "# HELP hailo_errors_total Total API errors.",
            "# TYPE hailo_errors_total counter",
            f"hailo_errors_total {snapshot['errors_total']}",
            "# HELP hailo_active_inferences Currently active inferences.",
            "# TYPE hailo_active_inferences gauge",
            f"hailo_active_inferences {snapshot['active_inferences']}",
        ]
        for endpoint, value in snapshot["endpoint_requests"].items():
            lines.append(f'hailo_endpoint_requests_total{{endpoint="{endpoint}"}} {value}')
        for endpoint, value in snapshot["endpoint_errors"].items():
            lines.append(f'hailo_endpoint_errors_total{{endpoint="{endpoint}"}} {value}')
        for detector, value in snapshot["detector_requests"].items():
            lines.append(f'hailo_detector_requests_total{{detector="{detector}"}} {value}')
        for name, values in (
            ("inference_latency_ms", snapshot["inference_latency_ms"]),
            ("queue_wait_ms", snapshot["queue_wait_ms"]),
        ):
            for key in ("p50", "p95", "p99"):
                lines.append(f"hailo_{name}_{key} {values[key]}")
        return "\n".join(lines) + "\n"


metrics = Metrics()


class InferenceQueue:
    def __init__(self, concurrency: int) -> None:
        self.semaphore = asyncio.Semaphore(concurrency)

    async def run(self, fn: Callable):
        queued_at = perf_counter()
        async with self.semaphore:
            queue_wait_ms = (perf_counter() - queued_at) * 1000
            metrics.active_inferences += 1
            try:
                return await fn(queue_wait_ms)
            finally:
                metrics.active_inferences -= 1


def require_api_key(settings: Settings):
    async def dependency(
        x_api_key: str | None = Header(default=None),
        x_rapidapi_proxy_secret: str | None = Header(default=None),
        x_rapidapi_user: str | None = Header(default=None),
        x_rapidapi_subscription: str | None = Header(default=None),
    ):
        if not settings.public_api_enabled:
            raise HTTPException(status_code=403, detail="Public API disabled")

        if settings.rapidapi_enabled:
            if not settings.rapidapi_proxy_secret:
                raise HTTPException(status_code=503, detail="RapidAPI proxy secret not configured")

            if x_rapidapi_proxy_secret == settings.rapidapi_proxy_secret:
                return {
                    "auth_type": "rapidapi",
                    "rapidapi_user": x_rapidapi_user,
                    "rapidapi_subscription": x_rapidapi_subscription,
                }

        if settings.api_keys:
            if x_api_key in settings.api_keys:
                return {
                    "auth_type": "direct",
                    "api_key": x_api_key,
                }

        raise HTTPException(status_code=401, detail="Invalid API credentials")

    return dependency
