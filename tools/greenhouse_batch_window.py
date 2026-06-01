#!/usr/bin/env python3
import argparse
import subprocess
import sys

import httpx


def post(client: httpx.Client, path: str) -> dict:
    response = client.post(path)
    response.raise_for_status()
    return response.json()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Load the greenhouse HEF, run a batch benchmark, then unload it."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:32168")
    parser.add_argument("--requests", type=int, default=30)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument(
        "--bench-script",
        default="tools/bench_service_interleaving.py",
        help="Path to the service benchmark helper.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the steps without loading or benchmarking.",
    )
    args = parser.parse_args()

    if args.dry_run:
        print(f"load -> {args.base_url}/v1/greenhouse/model/load")
        print(
            f"run -> {args.bench_script} --base-url {args.base_url} "
            f"--mode greenhouse --requests {args.requests} --concurrency {args.concurrency}"
        )
        print(f"unload -> {args.base_url}/v1/greenhouse/model/unload")
        return 0

    with httpx.Client(base_url=args.base_url, timeout=args.timeout) as client:
        try:
            load_body = post(client, "/v1/greenhouse/model/load")
            if not load_body.get("loaded"):
                print(f"failed to load greenhouse model: {load_body}", file=sys.stderr)
                return 1

            completed = subprocess.run(
                [
                    sys.executable,
                    args.bench_script,
                    "--base-url",
                    args.base_url,
                    "--mode",
                    "greenhouse",
                    "--requests",
                    str(args.requests),
                    "--concurrency",
                    str(args.concurrency),
                ],
                check=False,
            )
            return_code = completed.returncode
        finally:
            try:
                post(client, "/v1/greenhouse/model/unload")
            except Exception as exc:
                print(f"warning: failed to unload greenhouse model: {exc}", file=sys.stderr)

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
