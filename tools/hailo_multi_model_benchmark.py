#!/usr/bin/env python3
import argparse
from pathlib import Path
import shlex
import subprocess
import sys


def build_command(hefs: list[Path], seconds: int, json_path: Path | None) -> list[str]:
    command = [
        "hailortcli",
        "run2",
        "--time-to-run",
        str(seconds),
        "--mode",
        "full_async",
        "--scheduling-algorithm",
        "round_robin",
    ]
    if json_path:
        command.extend(["--json", str(json_path)])

    for hef in hefs:
        command.extend(["set-net", str(hef)])

    return command


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare or run a HailoRT single/multi-model benchmark."
    )
    parser.add_argument(
        "hef",
        nargs="+",
        type=Path,
        help="One or more HEF files. Pass the Frigate+ HEF plus a greenhouse HEF to compare.",
    )
    parser.add_argument("--seconds", type=int, default=10)
    parser.add_argument("--json", type=Path, help="Optional hailortcli JSON output path.")
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run the benchmark. Without this flag, only print the command.",
    )
    parser.add_argument(
        "--allow-service-contention",
        action="store_true",
        help="Acknowledge that running can contend with a live hailo-detectord service.",
    )
    args = parser.parse_args()

    missing = [hef for hef in args.hef if not hef.is_file()]
    if missing:
        for hef in missing:
            print(f"HEF does not exist: {hef}", file=sys.stderr)
        return 1

    command = build_command(args.hef, args.seconds, args.json)
    print(shlex.join(command))

    if not args.run:
        return 0

    if not args.allow_service_contention:
        print(
            "Refusing to run without --allow-service-contention. "
            "Stop hailo-detectord first or explicitly accept benchmark contention.",
            file=sys.stderr,
        )
        return 2

    completed = subprocess.run(command, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
