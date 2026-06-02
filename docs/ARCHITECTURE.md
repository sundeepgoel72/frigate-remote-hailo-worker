# Architecture

## Purpose

`frigate-remote-hailo-worker` is a remote inference worker for Frigate. Frigate remains the NVR, camera ingest system, motion detector, tracker, recorder, and event owner. This service runs separately on a Raspberry Pi 5 with a Hailo-8 accelerator and handles inference requests over HTTP.

## Target Deployment

```mermaid
flowchart LR
    CAM[IP Cameras] --> FRIGATE[HP400 / Frigate]
    FRIGATE -->|motion regions / crops| WORKER[RPi5 / hailo-detectord]
    WORKER --> HAILO[Hailo-8]
    WORKER --> FRIGATE
    FRIGATE --> HA[Home Assistant / MQTT]
```

## Responsibility Split

### Frigate / HP400

- RTSP camera ingest
- Video decode
- Motion detection
- Region generation
- Object tracking
- Event lifecycle
- Recording and snapshots
- Home Assistant and MQTT integration

### Hailo Worker / RPi5

- Object detection endpoint
- Face detection endpoint scaffold
- Face recognition API scaffold
- Optional debug capture
- Metrics and version endpoints
- Public API/RapidAPI-protected endpoints

## Object Detection Flow

```mermaid
sequenceDiagram
    participant F as Frigate
    participant W as Hailo Worker
    participant H as Hailo-8

    F->>W: POST /v1/vision/detection image crop
    W->>H: Run HEF inference
    H-->>W: NMS detections
    W-->>F: predictions[]
```

## Face Recognition Direction

Current face recognition uses deterministic development-only embeddings. This is suitable for API and workflow testing only.

Target flow:

```mermaid
flowchart TD
    P[Frigate person event] --> C[Snapshot or crop]
    C --> FD[Face detection]
    FD --> FE[Face embedding]
    FE --> MATCH[Local face library match]
    MATCH --> EVENT[Identity result / MQTT / HA]
```

Preferred production backend: InsightFace or ArcFace first, with optional Hailo embedding support later.

## Public API Layer

The `/public/...` endpoints are separate from LAN/internal Frigate endpoints and are protected by either direct API keys or RapidAPI provider headers.

```mermaid
flowchart LR
    USER[External client / RapidAPI] --> PROXY[HTTPS reverse proxy]
    PROXY --> WORKER[Public API endpoints]
    WORKER --> HAILO[Hailo-8]
```

## Current Maturity

Status: Alpha.

Core API architecture exists. Operational tooling, real face recognition, Hailo model compatibility hardening, installer safety, and live Frigate validation are still in progress.
