#!/usr/bin/env bash
set -euo pipefail

FRIGATE_CONTAINER="${FRIGATE_CONTAINER:-frigate}"
FRIGATE_CONFIG_DIR="${FRIGATE_CONFIG_DIR:-/mnt/ssd/frigate/config}"
WORKER_URL="${WORKER_URL:-http://192.168.1.175:32168/v1/vision/detection}"
API_TIMEOUT="${API_TIMEOUT:-2.0}"
MODEL_SOURCE="${MODEL_SOURCE:-}"
MODEL_METADATA_SOURCE="${MODEL_METADATA_SOURCE:-}"
MODEL_DEST_DIR="${MODEL_DEST_DIR:-$FRIGATE_CONFIG_DIR/remote-hailo}"
ONLY_CAMERA="${ONLY_CAMERA:-}"
APPLY="${APPLY:-false}"

usage() {
  cat <<'EOF'
Install or generate a Frigate remote-Hailo adapter config.

Environment variables:
  FRIGATE_CONTAINER       Docker container name. Default: frigate
  FRIGATE_CONFIG_DIR      Host path mounted as /config. Default: /mnt/ssd/frigate/config
  WORKER_URL              Remote worker detection URL.
  API_TIMEOUT             Frigate HTTP detector timeout seconds. Default: 2.0
  MODEL_SOURCE            Host path to the Hailo .hef model to copy.
  MODEL_METADATA_SOURCE   Host path to the matching Frigate model metadata .json.
  ONLY_CAMERA             Optional camera name to leave detect.enabled=true.
                          All other cameras get detect.enabled=false in the candidate.
  APPLY                   Set true to promote candidate config and restart Frigate.

Examples:
  MODEL_SOURCE=/tmp/model.hef MODEL_METADATA_SOURCE=/tmp/model.json ./deploy/install-frigate-adapter.sh
  ONLY_CAMERA=stairway APPLY=true ./deploy/install-frigate-adapter.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if ! docker inspect "$FRIGATE_CONTAINER" >/dev/null 2>&1; then
  echo "Frigate container '$FRIGATE_CONTAINER' was not found." >&2
  exit 1
fi

mkdir -p "$MODEL_DEST_DIR"

if [[ -n "$MODEL_SOURCE" ]]; then
  cp "$MODEL_SOURCE" "$MODEL_DEST_DIR/frigate-plus-hailo8.hef"
fi

if [[ -n "$MODEL_METADATA_SOURCE" ]]; then
  cp "$MODEL_METADATA_SOURCE" "$MODEL_DEST_DIR/frigate-plus-hailo8.json"
fi

if [[ ! -f "$MODEL_DEST_DIR/frigate-plus-hailo8.hef" ]]; then
  echo "Missing $MODEL_DEST_DIR/frigate-plus-hailo8.hef. Set MODEL_SOURCE or copy it first." >&2
  exit 1
fi

if [[ ! -f "$MODEL_DEST_DIR/frigate-plus-hailo8.json" ]]; then
  echo "Missing $MODEL_DEST_DIR/frigate-plus-hailo8.json. Set MODEL_METADATA_SOURCE or copy it first." >&2
  exit 1
fi

docker exec -i "$FRIGATE_CONTAINER" python3 - <<'PY'
import json
from pathlib import Path

base = Path("/config/remote-hailo")
metadata = json.loads((base / "frigate-plus-hailo8.json").read_text())
label_map = {int(key): value for key, value in metadata["labelMap"].items()}
with (base / "labelmap.txt").open("w") as label_file:
    for index in range(max(label_map) + 1):
        label_file.write(label_map.get(index, f"unknown_{index}") + "\n")
PY

helper="/tmp/create_frigate_remote_hailo_config.py"
cat >"$helper" <<'PY'
import os
from pathlib import Path

from ruamel.yaml import YAML

worker_url = os.environ["WORKER_URL"]
api_timeout = float(os.environ.get("API_TIMEOUT", "2.0"))
only_camera = os.environ.get("ONLY_CAMERA", "").strip()

yaml = YAML()
yaml.preserve_quotes = True
yaml.indent(mapping=2, sequence=4, offset=2)

source = Path("/config/config.yml")
candidate = Path("/config/config.remote-hailo.yml")
with source.open() as source_file:
    config = yaml.load(source_file)

config["detectors"] = {
    "remote_hailo8": {
        "type": "deepstack",
        "api_url": worker_url,
        "api_timeout": api_timeout,
    }
}

config["model"] = {
    "path": "/config/remote-hailo/frigate-plus-hailo8.hef",
    "labelmap_path": "/config/remote-hailo/labelmap.txt",
    "width": 640,
    "height": 640,
    "input_tensor": "nhwc",
    "input_pixel_format": "rgb",
    "input_dtype": "int",
    "model_type": "yolo-generic",
}

if only_camera:
    cameras = config.get("cameras", {})
    if only_camera not in cameras:
        raise SystemExit(f"ONLY_CAMERA={only_camera!r} is not in cameras: {', '.join(cameras)}")
    for name, camera in cameras.items():
        camera.setdefault("detect", {})["enabled"] = name == only_camera

with candidate.open("w") as candidate_file:
    yaml.dump(config, candidate_file)

print(candidate)
PY

docker cp "$helper" "$FRIGATE_CONTAINER:/tmp/create_frigate_remote_hailo_config.py"
docker exec \
  -e WORKER_URL="$WORKER_URL" \
  -e API_TIMEOUT="$API_TIMEOUT" \
  -e ONLY_CAMERA="$ONLY_CAMERA" \
  "$FRIGATE_CONTAINER" python3 /tmp/create_frigate_remote_hailo_config.py

validation_output="$(
  docker exec -e CONFIG_FILE=/config/config.remote-hailo.yml \
    "$FRIGATE_CONTAINER" python3 -m frigate --validate-config 2>&1
)"
echo "$validation_output"

if echo "$validation_output" | grep -q "Your config file is not valid"; then
  echo "Candidate config did not validate. Live config was not changed." >&2
  exit 1
fi

echo "Candidate written to $FRIGATE_CONFIG_DIR/config.remote-hailo.yml"

if [[ "$APPLY" == "true" ]]; then
  stamp="$(date +%Y%m%d-%H%M%S)"
  backup="$FRIGATE_CONFIG_DIR/config.yml.bak-remote-hailo-$stamp"
  cp "$FRIGATE_CONFIG_DIR/config.yml" "$backup"
  cp "$FRIGATE_CONFIG_DIR/config.remote-hailo.yml" "$FRIGATE_CONFIG_DIR/config.yml"
  docker restart "$FRIGATE_CONTAINER"
  echo "Applied. Backup: $backup"
else
  echo "Dry run complete. Set APPLY=true to promote and restart Frigate."
fi
