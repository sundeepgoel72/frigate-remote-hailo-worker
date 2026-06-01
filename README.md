# Frigate Remote Hailo Worker Prototype

This repo prototypes a LAN-local detector worker for this layout:

```text
Cameras -> HP400 Frigate -> HTTP detector calls -> Raspberry Pi 5 + Hailo8
```

Frigate remains the NVR, recorder, camera ingester, and motion-region generator. The Raspberry Pi only receives cropped detection frames from Frigate detector calls and returns object detections.

## Why This Shape

Frigate documents Hailo as a local detector that expects HailoRT inside the Frigate runtime/container. For remote offload without moving cameras or recording to the Pi, the prototype exposes a small HTTP service that can be configured as Frigate's existing DeepStack/CodeProject-style detector.

This keeps the first prototype narrow:

- No second Frigate instance on the Pi.
- No camera restreaming to the Pi.
- No full-frame continuous camera transport unless Frigate itself sends a detection crop.
- A mock backend is available for protocol testing before enabling Hailo.

## Repository Contents

- `src/hailo_detectord/` - FastAPI detector worker.
- `examples/frigate-hp400.yml` - Frigate detector config sample.
- `examples/rpi5-docker-compose.yml` - Pi-side worker deployment sample.
- `docs/api.md` - HTTP API contract.
- `docs/hailo-backend.md` - Notes for wiring real pyHailoRT/HailoRT inference.

## Quick Start On A Dev Machine

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -e ".[test]"
.\.venv\Scripts\hailo-detectord --host 0.0.0.0 --port 32168
```

Then check:

```powershell
curl http://localhost:32168/health
```

For a repeatable service smoke check:

```bash
python tools/smoke_hailo_detectord.py --base-url http://127.0.0.1:32168
python tools/smoke_hailo_detectord.py --image /path/to/crop.jpg
```

To run optional live Hailo integration tests on the Pi:

```bash
HAILO_INTEGRATION=1 pytest tests/test_hailo_integration.py -q
HAILO_INTEGRATION=1 HAILO_INTEGRATION_IMAGE=/path/to/crop.jpg pytest tests/test_hailo_integration.py -q
```

To prepare a HailoRT benchmark command for one or more HEF models:

```bash
python tools/hailo_multi_model_benchmark.py /opt/hailo-detectord/models/frigate-plus-hailo8.hef
```

To run a nightly greenhouse batch window without keeping the model resident all day:

```bash
python tools/greenhouse_batch_window.py --base-url http://127.0.0.1:32168
```

On the Pi, the systemd timers are configured to load the greenhouse HEF at 2:00 AM
and unload it at 2:10 AM local time.

## RPi5 Runtime Shape

Use the same Frigate-trained Hailo `.hef` model that worked in the local RPi5 Frigate test. Mount or copy that model onto the Pi and point `HAILO_MODEL_PATH` at it. If the model uses a custom Frigate label map, point `HAILO_LABELMAP_PATH` at the matching file too.

```bash
export HAILO_BACKEND=hailo
export HAILO_MODEL_PATH=/opt/hailo-detectord/models/frigate-plus-hailo8.hef
export HAILO_MODEL_METADATA_PATH=/opt/hailo-detectord/models/frigate-plus-hailo8.json
hailo-detectord --host 0.0.0.0 --port 32168
```

For HTTP/protocol testing without the accelerator, use `HAILO_BACKEND=mock`.

## Install As A Pi Service

On the RPi5, from this repo:

```bash
chmod +x deploy/install-rpi5-systemd.sh
./deploy/install-rpi5-systemd.sh
sudo nano /etc/hailo-detectord.env
sudo systemctl start hailo-detectord
curl http://127.0.0.1:32168/health
```

The installer creates a `hailo-detectord` systemd service listening on port `32168`.

## Frigate HP400 Config

Point Frigate at the Pi service using the existing HTTP detector path:

```yaml
detectors:
  remote_hailo8:
    type: deepstack
    api_url: http://rpi5-hailo.local:32168/v1/vision/detection
```

See `examples/frigate-hp400.yml` for a fuller sample.

## Self-Install Shape

RPi5 worker:

```bash
sudo bash deploy/install-rpi5-systemd.sh
```

Frigate host adapter:

```bash
APPLY=true ./deploy/install-frigate-adapter.sh
```

For a lower-risk first rollout, limit detection to one camera:

```bash
ONLY_CAMERA=stairway APPLY=true ./deploy/install-frigate-adapter.sh
```