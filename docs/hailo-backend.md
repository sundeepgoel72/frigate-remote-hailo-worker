# Hailo Backend Wiring Notes

`src/hailo_detectord/backends/hailo.py` loads a Hailo HEF through `hailo_platform` and expects the model to provide Hailo NMS-style detection output, which is the useful first target for Frigate-trained Hailo object models.

The backend needs these pieces for the selected HEF model:

1. Install matching HailoRT and `python3-hailort` on the RPi5.
2. Copy the Frigate-trained `.hef` file to the Pi.
3. Copy the matching Frigate label map if the model is custom.
4. Set `HAILO_BACKEND=hailo`.
5. Set `HAILO_MODEL_PATH=/path/to/model.hef`.
6. Set `HAILO_LABELMAP_PATH=/path/to/labelmap.txt` when needed.

The Frigate-side model dimensions in `examples/frigate-hp400.yml` must match the HEF input shape and postprocessor assumptions. For example, if the HEF is `yolov6n.hef` at `320x320`, keep Frigate's model width and height at `320`.

## Validation Order

1. Run the worker with `HAILO_BACKEND=mock`.
2. Configure HP400 Frigate to call the Pi endpoint.
3. Confirm Frigate detector requests reach the Pi.
4. Switch `HAILO_BACKEND=hailo`.
5. Compare detections against the known-good local RPi5 Frigate + Hailo setup.

## Model Compatibility

The first implementation expects object detector outputs shaped like Hailo NMS:

- one detection set per class
- each detection has at least `[y_min, x_min, y_max, x_max, score]` by default
- normalized boxes are converted back to pixels for the submitted crop

If the Frigate-trained model returns `[x_min, y_min, x_max, y_max, score]`, set:

```bash
export HAILO_BBOX_ORDER=xyxy
```

## Multi-Model Hailo Benchmarking

HailoRT can schedule multiple HEF models on one Hailo-8 device, but throughput is
shared. Benchmark before running a greenhouse model beside the Frigate+ detector.

Print the single-model benchmark command:

```bash
python tools/hailo_multi_model_benchmark.py \
  /opt/hailo-detectord/models/frigate-plus-hailo8.hef
```

Print the multi-model benchmark command once a greenhouse HEF exists:

```bash
python tools/hailo_multi_model_benchmark.py \
  /opt/hailo-detectord/models/frigate-plus-hailo8.hef \
  /opt/hailo-detectord/models/greenhouse-disease.hef
```

To actually run the benchmark:

```bash
sudo systemctl stop hailo-detectord
python tools/hailo_multi_model_benchmark.py --run --allow-service-contention \
  /opt/hailo-detectord/models/frigate-plus-hailo8.hef
sudo systemctl start hailo-detectord
```

Use `--allow-service-contention` only when the service is stopped or when
temporary detector latency impact is acceptable. The helper defaults to printing
the command instead of running it.

## Optional Greenhouse HEF

The greenhouse endpoint uses the color baseline by default. To load a second HEF
beside the Frigate detector in the same process:

```bash
export HAILO_GREENHOUSE_BACKEND=hailo
export HAILO_GREENHOUSE_MODEL_PATH=/opt/hailo-detectord/models/greenhouse-disease.hef
export HAILO_GREENHOUSE_LABELMAP_PATH=/opt/hailo-detectord/models/greenhouse-labels.txt
```

For stand-in testing with the bundled ResNet classifier:

```bash
export HAILO_GREENHOUSE_BACKEND=hailo
export HAILO_GREENHOUSE_MODEL_PATH=/usr/share/hailo-models/resnet_v1_50_h8l.hef
```

When both models are configured, `/version` reports
`greenhouse_model_loaded: true`.

By default the greenhouse HEF is not loaded at startup. Use the lifecycle
endpoints to batch it in only for the night run:

```bash
curl -X POST http://127.0.0.1:32168/v1/greenhouse/model/load
python tools/greenhouse_batch_window.py --base-url http://127.0.0.1:32168
curl -X POST http://127.0.0.1:32168/v1/greenhouse/model/unload
```

If you want the greenhouse HEF resident at boot, set:

```bash
export HAILO_GREENHOUSE_AUTOLOAD=true
```

To batch the greenhouse model only between 2:00 and 2:10 local time, install the
timer units from `deploy/` and let the load/unload services run automatically.
The current schedule is:

```text
02:00  load greenhouse HEF
02:10  unload greenhouse HEF
```

The installer enables both timers. You can verify them with:

```bash
systemctl list-timers | grep hailo-greenhouse
```