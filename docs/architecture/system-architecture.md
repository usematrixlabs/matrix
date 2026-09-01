# Matrix — System Architecture

This document describes the canonical system architecture, subsystem boundaries, data flow, and interface contracts for **Matrix** (SIH26158).

---

## 1. System Overview

**Matrix** is a UAV video-to-3D geospatial reconstruction system that processes single-pass UAV video along with optional telemetry, GPS, IMU, and camera calibration into a georeferenced, validated 3D representation.

The five primary subsystems are:

| Subsystem | Name | Responsibility | Output Contract |
| :--- | :--- | :--- | :--- |
| **S1** | Visual Perception | Ingests video, extracts frames, assigns stable IDs, timestamps, quality scores, keyframe flags, and packages `s1_output/` | `docs/architecture/contracts/perception-localization.md` |
| **S2** | Localization & Sensor Fusion | Estimates camera poses, trajectory, and fuses telemetry | `docs/architecture/contracts/` |
| **S3** | 3D Reconstruction | Generates dense point clouds, meshes, and textures | `docs/architecture/contracts/` |
| **S4** | Georeferencing & Validation | Performs world coordinate transformation and accuracy validation | `docs/architecture/contracts/` |
| **S5** | Application & Deployment | UI, API orchestration, and visualization | `docs/architecture/contracts/` |

---

## 2. S1 Packaging & Contract (`s1_output/`)

S1 packages visual observations into a self-contained, portable directory structure consumed by S2:

```text
s1_output/
│
├── frames/
│   ├── frame_000001.jpg
│   ├── frame_000002.jpg
│   └── ...
│
└── observations.json
```

### Observation Item Schema (`observations.json`)

```json
{
  "observation_id": "frame_000123",
  "timestamp": 12.34,
  "image": "frames/frame_000123.jpg",
  "camera": {
    "width": 1920,
    "height": 1080,
    "intrinsics": null,
    "distortion": null
  },
  "quality": {
    "status": "GOOD",
    "blur_score": 245.3,
    "quality_score": 88.5,
    "flags": []
  },
  "keyframe": true
}
```

### Key Architectural Principles

1. **Non-Destructive Observation Retention:** S1 preserves all valid candidate observations in `observations.json`, marking keyframe status with `keyframe: bool`. S2 decides whether to use all observations or keyframes only.
2. **Stable Deterministic Identifiers:** Observations have immutable IDs (`frame_000001`, `frame_000002`, ...) that persist across S1 $\rightarrow$ S2 $\rightarrow$ S3.
3. **Capture Timestamps:** Timestamps represent source video capture time in seconds ($t_k < t_{k+1}$ monotonic).
4. **Camera Intrinsics:** Dimensions (`width`, `height`) are always known. Intrinsics (`fx, fy, cx, cy`) and distortion are preserved when supplied, and explicitly set to `null` with `is_calibrated: false` when unavailable.
5. **Portable Relative Paths:** Image paths inside `observations.json` use relative paths (`frames/frame_000001.jpg`) from the package root.

---

## 3. End-to-End Pipeline Sequence

```text
                         UAV INPUT (Video + Telemetry)
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │ S1 · Visual Perception    │
                        │                           │
                        │ • Ingest & validate video │
                        │ • Fixed-interval sampling │
                        │ • Stable ID generation    │
                        │ • Monotonic timestamps    │
                        │ • Multi-metric quality    │
                        │ • Keyframe detection      │
                        │ • Camera calibration      │
                        │ • Portable packaging      │
                        └─────────────┬─────────────┘
                                      │ s1_output/ (frames/ + observations.json)
                                      ▼
                        ┌───────────────────────────┐
                        │ S2 · Localization         │
                        │    & Sensor Fusion        │
                        └─────────────┬─────────────┘
                                      │ S1 observations + Poses
                                      ▼
                        ┌───────────────────────────┐
                        │ S3 · 3D Reconstruction    │
                        └─────────────┬─────────────┘
                                      │ 3D Point Cloud / Mesh
                                      ▼
                        ┌───────────────────────────┐
                        │ S4 · Georeferencing       │
                        │    & Validation           │
                        └─────────────┬─────────────┘
                                      │ Georeferenced Scene
                                      ▼
                        ┌───────────────────────────┐
                        │ S5 · Application          │
                        │    & Visualization        │
                        └───────────────────────────┘
```

