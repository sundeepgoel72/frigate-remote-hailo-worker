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

## Coordinate Contract

Bounding boxes are pixel coordinates relative to the submitted image, not the original camera frame. Frigate is expected to map detector results back to its motion region internally when using its detector pipeline.
