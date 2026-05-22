# Camera-Level Testing

Frigate 0.17.1 does not support selecting an object detector per camera.

I verified this directly against the HP400 container by adding:

```yaml
cameras:
  stairway:
    detect:
      detector: remote_hailo8
```

Frigate rejected it:

```text
Key     : cameras -> stairway -> detect -> detector
Value   : remote_hailo8
Message : Extra inputs are not permitted
```

## Practical Derisking Pattern

Detector choice is global, but detection can be limited per camera. For a safer test, generate a config that uses the remote detector globally while setting `detect.enabled: false` on every camera except one.

Using the installer:

```bash
ONLY_CAMERA=stairway APPLY=true ./deploy/install-frigate-adapter.sh
```

That still creates a global Frigate detector process named `remote_hailo8`, but only the selected camera sends object detection work to it.

To re-enable all cameras, rerun without `ONLY_CAMERA`:

```bash
APPLY=true ./deploy/install-frigate-adapter.sh
```
