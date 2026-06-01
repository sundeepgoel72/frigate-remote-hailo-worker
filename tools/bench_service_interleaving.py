#!/usr/bin/env python3
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from time import perf_counter

import cv2
import httpx
import numpy as np


@dataclass
class Result:
    endpoint: str
    status_code: int
    latency_ms: float
    backend: str | None
    error: str | None = None


def jpeg() -> bytes:
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    image[:, :] = (28, 112, 48)
    cv2.rectangle(image, (170, 140), (470, 520), (35, 180, 70), -1)
    cv2.circle(image, (320, 330), 70, (18, 95, 165), -1)
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("failed to encode synthetic JPEG")
    return encoded.tobytes()


def post_image(base_url: str, endpoint: str, image: bytes, timeout: float) -> Result:
    started = perf_counter()
    try:
        with httpx.Client(base_url=base_url, timeout=timeout) as client:
            response = client.post(
                endpoint,
                files={"image": ("frame.jpg", image, "image/jpeg")},
            )
        elapsed_ms = (perf_counter() - started) * 1000
        backend = None
        try:
            backend = response.json().get("backend")
        except ValueError:
            pass
        return Result(endpoint, response.status_code, elapsed_ms, backend)
    except Exception as exc:
        elapsed_ms = (perf_counter() - started) * 1000
        return Result(endpoint, 0, elapsed_ms, None, str(exc))


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = min(len(values) - 1, round((pct / 100) * (len(values) - 1)))
    return values[index]


def summarize(name: str, results: list[Result]) -> None:
    latencies = [result.latency_ms for result in results]
    errors = [result for result in results if result.status_code != 200 or result.error]
    backends = sorted({result.backend for result in results if result.backend})
    print(
        f"{name}: requests={len(results)} errors={len(errors)} "
        f"p50_ms={percentile(latencies, 50):.2f} "
        f"p95_ms={percentile(latencies, 95):.2f} "
        f"backends={','.join(backends) if backends else '-'}"
    )
    for error in errors[:5]:
        print(
            f"  error endpoint={error.endpoint} status={error.status_code} "
            f"latency_ms={error.latency_ms:.2f} detail={error.error or ''}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark service-level Hailo endpoint interleaving.")
    parser.add_argument("--base-url", default="http://127.0.0.1:32169")
    parser.add_argument("--requests", type=int, default=30)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument(
        "--mode",
        choices=("detect", "greenhouse", "mixed"),
        required=True,
    )
    args = parser.parse_args()

    image = jpeg()
    endpoints = []
    if args.mode in ("detect", "mixed"):
        endpoints.extend(["/v1/vision/detection"] * args.requests)
    if args.mode in ("greenhouse", "mixed"):
        endpoints.extend(["/v1/greenhouse/disease/classify"] * args.requests)

    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(post_image, args.base_url, endpoint, image, args.timeout)
            for endpoint in endpoints
        ]
        for future in as_completed(futures):
            results.append(future.result())

    summarize(args.mode, results)
    return 1 if any(result.status_code != 200 or result.error for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
