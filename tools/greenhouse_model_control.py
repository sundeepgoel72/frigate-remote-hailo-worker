#!/usr/bin/env python3
import argparse
import sys

import httpx


def main() -> int:
    parser = argparse.ArgumentParser(description="Control the greenhouse HEF lifecycle.")
    parser.add_argument("--base-url", default="http://127.0.0.1:32168")
    parser.add_argument(
        "action",
        choices=("status", "load", "unload"),
        help="Greenhouse model lifecycle action.",
    )
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    path = {
        "status": "/v1/greenhouse/model/status",
        "load": "/v1/greenhouse/model/load",
        "unload": "/v1/greenhouse/model/unload",
    }[args.action]

    method = "GET" if args.action == "status" else "POST"

    with httpx.Client(base_url=args.base_url, timeout=args.timeout) as client:
        response = client.request(method, path)
        if response.status_code >= 400:
            print(response.text, file=sys.stderr)
            return 1
        print(response.text)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
