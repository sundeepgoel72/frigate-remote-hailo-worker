#!/usr/bin/env python3
import argparse
import json
import mimetypes
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen
from uuid import uuid4


def get_json(base_url: str, path: str, timeout: float) -> dict:
    with urlopen(urljoin(base_url, path), timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def post_image(base_url: str, path: str, image_path: Path, timeout: float) -> dict:
    boundary = f"----hailo-detectord-smoke-{uuid4().hex}"
    content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    image_data = image_path.read_bytes()

    body = b"".join(
        [
            f"--{boundary}\r\n".encode("ascii"),
            (
                'Content-Disposition: form-data; name="image"; '
                f'filename="{image_path.name}"\r\n'
            ).encode("utf-8"),
            f"Content-Type: {content_type}\r\n\r\n".encode("ascii"),
            image_data,
            f"\r\n--{boundary}--\r\n".encode("ascii"),
        ]
    )

    request = Request(
        urljoin(base_url, path),
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def print_check(name: str, detail: str) -> None:
    print(f"ok {name}: {detail}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test a running hailo-detectord service.")
    parser.add_argument("--base-url", default="http://127.0.0.1:32168")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument(
        "--image",
        type=Path,
        help="Optional JPEG/PNG crop to send to /v1/vision/detection.",
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/") + "/"

    try:
        health = get_json(base_url, "/health", args.timeout)
        if health.get("status") != "ok":
            print(f"health failed: {health}", file=sys.stderr)
            return 1
        print_check("health", f"backend={health.get('backend')}")

        version = get_json(base_url, "/version", args.timeout)
        print_check(
            "version",
            f"app_version={version.get('app_version')} model_id={version.get('model_id')}",
        )

        if args.image:
            if not args.image.is_file():
                print(f"image does not exist: {args.image}", file=sys.stderr)
                return 1

            detection = post_image(
                base_url,
                "/v1/vision/detection",
                args.image,
                args.timeout,
            )
            if detection.get("success") is not True:
                print(f"detection failed: {detection}", file=sys.stderr)
                return 1
            print_check(
                "detection",
                (
                    f"backend={detection.get('backend')} "
                    f"predictions={len(detection.get('predictions', []))}"
                ),
            )
    except HTTPError as exc:
        print(f"HTTP {exc.code} from {exc.url}: {exc.reason}", file=sys.stderr)
        return 1
    except (OSError, URLError, TimeoutError) as exc:
        print(f"request failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
