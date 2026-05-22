# Frigate Adapter

The HP400 Frigate adapter uses Frigate's built-in `deepstack` detector plugin as an HTTP client for the RPi5 worker.

The live HP400 config was changed to:

```yaml
detectors:
  remote_hailo8:
    type: deepstack
    api_url: http://192.168.1.175:32168/v1/vision/detection
    api_timeout: 2.0

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

The local `model.path` is present so Frigate validation and model hashing have a real file. The DeepStack detector does not run that model locally; it sends detector crops to the Pi service.

## Live Files On HP400

- `/mnt/ssd/frigate/config/config.yml`
- `/mnt/ssd/frigate/config/config.remote-hailo.yml`
- `/mnt/ssd/frigate/config/remote-hailo/frigate-plus-hailo8.hef`
- `/mnt/ssd/frigate/config/remote-hailo/frigate-plus-hailo8.json`
- `/mnt/ssd/frigate/config/remote-hailo/labelmap.txt`

## Rollback

The previous config was backed up as:

```bash
/mnt/ssd/frigate/config/config.yml.bak-remote-hailo-
```

To roll back:

```bash
sudo cp /mnt/ssd/frigate/config/config.yml.bak-remote-hailo- /mnt/ssd/frigate/config/config.yml
docker restart frigate
```

## One-Camera Test Mode

Frigate does not support detector selection per camera, but you can limit testing by enabling detection on only one camera while using the remote detector globally:

```bash
ONLY_CAMERA=stairway APPLY=true ./deploy/install-frigate-adapter.sh
```

See `docs/camera-level-testing.md`.

## Verification

Frigate stats should show:

```json
"detectors": {
  "remote_hailo8": {
    "inference_speed": 41.27
  }
}
```

The Pi should show HTTP calls from HP400:

```bash
journalctl -u hailo-detectord --since "1 minute ago"
```
