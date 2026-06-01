# HTTP API

The worker exposes two detection endpoints.

## `GET /health`

Returns backend status:

```json
{
  "status": "ok",
  "backend": "mock",
  "model_path": null
}
```

## `POST /v1/vision/detection`

DeepStack-shaped endpoint intended for Frigate's HTTP detector configuration.

Request:

- `multipart/form-data`
- file field: `image`
- value: JPEG/PNG crop sent by Frigate

Response:

```json
{
  "success": true,
  "predictions": [
    {
      "label": "person",
      "confidence": 0.8,
      "x_min": 80,
      "y_min": 64,
      "x_max": 240,
      "y_max": 256
    }
  ],
  "inference_ms": 2.1,
  "backend": "mock"
}
```

## `POST /detect`

Raw-image endpoint for direct tests and future custom adapters.

Request body is the encoded image bytes. Response is the same as `/v1/vision/detection`.

## `POST /v1/greenhouse/disease/classify`

Prototype greenhouse crop-health classifier endpoint.

Request:

- `multipart/form-data`
- file field: `image`
- value: JPEG/PNG leaf or crop image

Response:

```json
{
  "success": true,
  "predictions": [
    {"label": "healthy", "confidence": 0.82},
    {"label": "chlorosis_yellowing", "confidence": 0.21}
  ],
  "inference_ms": 0.7,
  "backend": "greenhouse-color-baseline",
  "warning": "Baseline color-health classifier for greenhouse API validation; not a production disease diagnosis model."
}
```

This endpoint is intentionally a baseline classifier while the real crop disease
model is selected, calibrated, and converted for the target runtime.

## `GET /v1/greenhouse/model/status`

Returns whether the greenhouse Hailo HEF is currently loaded.

## `POST /v1/greenhouse/model/load`

Loads the greenhouse HEF into the shared Hailo runtime.

## `POST /v1/greenhouse/model/unload`

Unloads the greenhouse HEF so Frigate can keep the chip for the object model.

## Coordinate Contract

Bounding boxes are pixel coordinates relative to the submitted image, not the original camera frame. Frigate is expected to map detector results back to its motion region internally when using its detector pipeline.
