# Known Issues

## Face Recognition

Current face embeddings are deterministic development scaffolding and are not production-grade recognition.

Impact:
- Useful for API testing.
- Useful for Double Take integration testing.
- Not suitable for identity accuracy testing.

Planned fix:
- InsightFace backend.
- ArcFace backend.
- Future Hailo embedding backend.

## HEF Compatibility

The current implementation assumes Frigate-style object detection models producing NMS-like outputs.

Impact:
- Some custom HEFs may not work.
- Classification models are not supported.
- Unsupported output tensors may fail with limited diagnostics.

Planned fix:
- HEF output introspection.
- Tensor shape validation.
- Clear compatibility errors.

## Load Testing

The worker has not yet completed long-duration validation under Frigate production traffic.

Unknowns:
- Queue growth.
- Long-term Hailo stability.
- Sustained throughput.

## Installer Safety

Current installer tooling is functional but not yet production-hardened.

Missing:
- Dry-run mode.
- Rollback mode.
- Config diff preview.
- Automatic Frigate discovery.

## Documentation

Postman examples and Frigate validation guides are still being expanded.
