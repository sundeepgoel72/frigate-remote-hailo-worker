# Handoff Context

## Goal

Prototype remote Hailo8 inference for Frigate:

- HP400 x86 mini PC keeps running Frigate as the only NVR, recorder, camera ingester, tracker, and UI.
- Raspberry Pi 5 at `192.168.1.175` runs only a Hailo8 inference worker.
- Frigate sends detector crops over LAN HTTP.
- The Pi returns object detections.
- No second Frigate instance is required on the Pi.
- No camera restreaming to the Pi is required.

## Live Deployment

### RPi5 Worker

Host:

```text
rpi-ai / 192.168.1.175
```

Service:

```text
hailo-detectord.service
```

Status at handoff:

```text
enabled
active
```

Current deployed package:

```text
frigate-remote-hailo-worker 0.7.0
```

Health endpoint:

```text
http://192.168.1.175:32168/health
```

Detector endpoint:

```text
http://192.168.1.175:32168/v1/vision/detection
```

Version endpoint:

```text
http://192.168.1.175:32168/version
```

Runtime env:

```bash
HAILO_BACKEND=hailo
HAILO_MODEL_PATH=/opt/hailo-detectord/models/frigate-plus-hailo8.hef
HAILO_MODEL_METADATA_PATH=/opt/hailo-detectord/models/frigate-plus-hailo8.json
HAILO_CONFIDENCE_THRESHOLD=0.35
HAILO_BBOX_ORDER=yxyx
HAILO_INPUT_PIXEL_FORMAT=rgb
```

Current `/version` summary after the 0.7.0 service update:

```text
app_version: 0.7.0
backend: hailo
model_id: yolov9s
input_shape: [640, 640, 3]
label_count: 41
hailort_available: true
greenhouse_backend: color
greenhouse_model_loaded: false
```

Model source:

```text
/home/frigate/config/model_cache/7dc8d0e06ea0ec501934d46b3d2f09bf
```

This cached Frigate+ model metadata says:

```text
name: yolov9s
hailoDevice: hailo8
supportedDetectors: ["hailo8l"]
width: 640
height: 640
inputShape: nhwc
inputDataType: int
pixelFormat: rgb
```

### HP400 Frigate Adapter

Host:

```text
hp400 / 192.168.1.72
```

Frigate container:

```text
frigate
```

Adapter uses Frigate's built-in `deepstack` detector plugin as the HTTP client:

```yaml
detectors:
  remote_hailo8:
    type: deepstack
    api_url: http://192.168.1.175:32168/v1/vision/detection
    api_timeout: 2.0
```

Frigate model config is manually matched to the Hailo model:

```yaml
model:
  path: /config/remote-hailo/frigate-plus-hailo8.hef
  labelmap_path: /config/remote-hailo/labelmap.txt
  width: 640
  height: 640
  input_tensor: nhwc
  input_pixel_format: rgb
  input_dtype: int
  model_type: yolo-generic
```

HP400 model adapter files:

```text
/mnt/ssd/frigate/config/remote-hailo/frigate-plus-hailo8.hef
/mnt/ssd/frigate/config/remote-hailo/frigate-plus-hailo8.json
/mnt/ssd/frigate/config/remote-hailo/labelmap.txt
```

## Verification Already Completed

RPi5:

- `hailortcli` sees `/dev/hailo0`.
- `hailo_platform` imports successfully.
- Worker service starts with backend `hailo`.
- Worker service was updated and restarted with package version `0.7.0`.
- Synthetic JPEG request returns HTTP 200.
- Real Frigate-origin WebP request returns HTTP 200.
- Empty upload returns clean HTTP 400.
- Local regression suite passes: `16 passed, 2 skipped`.
- Live integration metadata test passes against `http://127.0.0.1:32168`.
- Live smoke test reports `ok health: backend=hailo` and `app_version=0.7.0`.

HP400:

- Frigate config candidate validates with `python3 -m frigate --validate-config`.
- Frigate restarts and becomes healthy.
- Frigate stats show:

```json
"detectors": {
  "remote_hailo8": {
    "inference_speed": 41.27
  }
}
```

- RPi5 worker logs show many successful detector calls from `192.168.1.72`.

## Camera-Level Testing Finding

Frigate 0.17.1 does not support selecting a detector per camera.

Validated rejected config:

```yaml
cameras:
  stairway:
    detect:
      detector: remote_hailo8
```

Frigate validation error:

```text
Key     : cameras -> stairway -> detect -> detector
Value   : remote_hailo8
Message : Extra inputs are not permitted
```

Safe test approach:

- Detector remains global.
- Enable detection on only one camera.
- Disable `detect.enabled` on all other cameras.

Installer supports:

```bash
ONLY_CAMERA=stairway APPLY=true ./deploy/install-frigate-adapter.sh
```

## Greenhouse Batch Scheduling

Release `0.7.0` adds optional greenhouse crop-health inference without keeping a
second HEF resident on the Hailo-8 throughout the day.

Implementation shape:

- Frigate object HEF remains the primary resident model.
- Greenhouse HEF is lazy-loaded through lifecycle endpoints.
- `POST /v1/greenhouse/disease/classify` returns `503` when
  `HAILO_GREENHOUSE_BACKEND=hailo` but the greenhouse model is not loaded.
- Default greenhouse mode remains the color baseline unless
  `HAILO_GREENHOUSE_BACKEND=hailo` is configured.

Lifecycle endpoints:

```text
GET  /v1/greenhouse/model/status
POST /v1/greenhouse/model/load
POST /v1/greenhouse/model/unload
POST /v1/greenhouse/disease/classify
```

Installed timers:

```text
hailo-greenhouse-load.timer    Tue 2026-06-02 02:00:00 IST
hailo-greenhouse-unload.timer  Tue 2026-06-02 02:10:00 IST
```

Both timers are enabled and active. The installer now uses
`systemctl enable --now` for both greenhouse timers so future installs start
the scheduled jobs immediately.

Greenhouse status after deployment:

```json
{"success": true, "loaded": false, "model_path": null, "backend": "color"}
```

Details and usage:

```text
docs/greenhouse-batch-scheduling.md
```

Hailo-8 benchmark finding:

- Frigate+ HEF alone: about `43.48 FPS`.
- Frigate+ plus ResNet stand-in loaded together: Frigate+ about `19.98 FPS`,
  ResNet about `24.97 FPS`.
- Decision: do not keep both models loaded all day; use the 2:00 to 2:10 AM
  batch window for greenhouse jobs.

## Rollback

Previous HP400 Frigate config backup:

```text
/mnt/ssd/frigate/config/config.yml.bak-remote-hailo-
```

Rollback:

```bash
sudo cp /mnt/ssd/frigate/config/config.yml.bak-remote-hailo- /mnt/ssd/frigate/config/config.yml
docker restart frigate
```

RPi service stop:

```bash
sudo systemctl stop hailo-detectord
sudo systemctl disable hailo-detectord
```

## Repo State

GitHub:

```text
https://github.com/sundeepgoel72/frigate-remote-hailo-worker
```

Visibility:

```text
private
```

Main branch:

```text
master
```

Important files:

```text
src/hailo_detectord/              Worker service
deploy/install-rpi5-systemd.sh    Pi installer
deploy/install-frigate-adapter.sh HP400 Frigate adapter installer
docs/frigate-adapter.md           Adapter details
docs/camera-level-testing.md      Camera-level test finding
docs/greenhouse-batch-scheduling.md Greenhouse lifecycle and batch usage
docs/hailo-backend.md             Hailo backend notes
examples/frigate-hp400.yml        Frigate config example
```

Latest release:

```text
v0.7.0
```

## Known Notes

- The HP400's current active Frigate+ OpenVINO model id was not Hailo-compatible.
- The usable Hailo-compatible Frigate+ model was found in the old RPi Frigate cache: `7dc8d0e06ea0ec501934d46b3d2f09bf`.
- The worker pins `numpy<2` and `opencv-python-headless<4.10`; this mattered for HailoRT buffer compatibility.
- The current adapter uses Frigate's DeepStack HTTP detector plugin because it already converts detector tensors to images and parses the same prediction JSON shape.

## Suggested Next Steps

1. Configure `HAILO_GREENHOUSE_BACKEND=hailo` and
   `HAILO_GREENHOUSE_MODEL_PATH` once the selected greenhouse disease HEF is
   available.
2. Run the 2:00 to 2:10 AM batch window with real greenhouse images and review
   latency before expanding the window.
3. Improve installer prompts and auto-discovery of Frigate config path.
4. Add optional systemd hardening after the prototype stabilizes.
5. Decide whether to keep using the DeepStack-shaped protocol or add a Frigate-native custom detector plugin later.
