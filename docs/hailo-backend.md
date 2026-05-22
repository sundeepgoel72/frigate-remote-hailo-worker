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
