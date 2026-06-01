# Greenhouse Batch Scheduling

This release adds optional greenhouse crop-health inference without keeping the
second HEF resident on the Hailo-8 all day. The Frigate object detector remains
the primary resident model. The greenhouse model is loaded only for a short
nightly batch window, then unloaded so normal Frigate detector latency is not
penalized outside that window.

## Implementation

The Hailo backend now owns one shared `VDevice` configured with round-robin
scheduling. The Frigate object detector is configured during service startup.
The greenhouse classifier is configured lazily through lifecycle endpoints:

- `GET /v1/greenhouse/model/status`
- `POST /v1/greenhouse/model/load`
- `POST /v1/greenhouse/model/unload`

When `HAILO_GREENHOUSE_BACKEND=hailo` and the greenhouse model is not loaded,
`POST /v1/greenhouse/disease/classify` returns `503` instead of trying to load
the model implicitly. This keeps accidental daytime requests from reducing
Frigate throughput.

The default greenhouse path remains the lightweight color baseline, which is
useful for API wiring tests only. It is not a production plant disease model.

## Runtime Configuration

Set the normal Frigate object model as before:

```bash
HAILO_BACKEND=hailo
HAILO_MODEL_PATH=/opt/hailo-detectord/models/frigate-plus-hailo8.hef
HAILO_MODEL_METADATA_PATH=/opt/hailo-detectord/models/frigate-plus-hailo8.json
```

To enable a greenhouse HEF for scheduled batch use:

```bash
HAILO_GREENHOUSE_BACKEND=hailo
HAILO_GREENHOUSE_MODEL_PATH=/opt/hailo-detectord/models/greenhouse-disease.hef
HAILO_GREENHOUSE_LABELMAP_PATH=/opt/hailo-detectord/models/greenhouse-labels.txt
HAILO_GREENHOUSE_AUTOLOAD=false
```

For stand-in testing before the real greenhouse model exists, the bundled ResNet
HEF can be used:

```bash
HAILO_GREENHOUSE_MODEL_PATH=/usr/share/hailo-models/resnet_v1_50_h8l.hef
```

## Nightly Window

The deployment installs two timers:

```text
02:00  hailo-greenhouse-load.timer
02:10  hailo-greenhouse-unload.timer
```

The load timer calls:

```bash
/opt/hailo-detectord/venv/bin/python \
  /opt/hailo-detectord/app/tools/greenhouse_model_control.py \
  --base-url http://127.0.0.1:32168 load
```

The unload timer calls the same helper with `unload`.

Verify the timers:

```bash
systemctl list-timers | grep hailo-greenhouse
systemctl status hailo-greenhouse-load.timer
systemctl status hailo-greenhouse-unload.timer
```

## Manual Operation

Load the greenhouse HEF:

```bash
python tools/greenhouse_model_control.py --base-url http://127.0.0.1:32168 load
```

Run the greenhouse batch helper:

```bash
python tools/greenhouse_batch_window.py --base-url http://127.0.0.1:32168
```

Unload after the batch:

```bash
python tools/greenhouse_model_control.py --base-url http://127.0.0.1:32168 unload
```

## Regression Checks

Local tests:

```bash
pytest -q
```

Live service smoke:

```bash
python tools/smoke_hailo_detectord.py --base-url http://127.0.0.1:32168
```

Optional live integration tests:

```bash
HAILO_INTEGRATION=1 pytest tests/test_hailo_integration.py -q
HAILO_INTEGRATION=1 HAILO_INTEGRATION_IMAGE=/path/to/crop.jpg \
  pytest tests/test_hailo_integration.py -q
```

Service-level interleaving benchmark:

```bash
python tools/bench_service_interleaving.py \
  --base-url http://127.0.0.1:32168 --mode detect
python tools/bench_service_interleaving.py \
  --base-url http://127.0.0.1:32168 --mode greenhouse
python tools/bench_service_interleaving.py \
  --base-url http://127.0.0.1:32168 --mode mixed
```

## Observed Hailo-8 Results

Using the Frigate+ HEF and a ResNet HEF as the greenhouse stand-in:

- Frigate+ model alone through `hailortcli run2`: about `43.48 FPS`.
- Two models resident through `hailortcli run2`: Frigate+ about `19.98 FPS`,
  ResNet about `24.97 FPS`.
- Service-level interleaving with both models loaded completed without request
  errors, with mixed requests showing higher latency as expected.

The chosen production behavior is therefore scheduled lazy loading: keep the
Frigate model resident all day, load the greenhouse model only for the 2:00 to
2:10 AM batch window, then unload it.
