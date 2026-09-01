# S1 → S2 Interface Contract

## Overview

S1 (Visual Perception) produces structured visual observations, frame images, camera calibration, degradation diagnostics, and available UAV sensor information for S2 (Localization & Sensor Fusion) to estimate camera trajectory and poses.

---

## 1. Physical Package Structure (`s1_output/`)

S1 packages observations into a self-contained, portable artifact directory:

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

Image references within `observations.json` use relative paths (`frames/frame_000001.jpg`) from the package root.

---

## 2. Observation Schema (`observations.json`)

The canonical schema is formally defined in [`observations-schema.json`](observations-schema.json).

### Root Container

```json
{
  "schema_version": "1.0.0",
  "subsystem": "S1_Visual_Perception",
  "created_at": "2026-09-01T15:30:00Z",
  "video_source": "data/raw/flight_01.mp4",
  "total_observations": 120,
  "keyframe_count": 24,
  "keyframe_density": 0.20,
  "temporal_information": {
    "time_unit": "seconds",
    "time_reference": "relative_capture_time",
    "is_monotonic": true,
    "duration_seconds": 60.0
  },
  "camera": {
    "width": 1920,
    "height": 1080,
    "intrinsics": {
      "fx": 1450.0,
      "fy": 1452.0,
      "cx": 960.0,
      "cy": 540.0,
      "camera_matrix": [
        [1450.0, 0.0, 960.0],
        [0.0, 1452.0, 540.0],
        [0.0, 0.0, 1.0]
      ]
    },
    "distortion": {
      "coefficients": [-0.12, 0.05, 0.0, 0.0, 0.0],
      "model": "radtan"
    },
    "is_calibrated": true
  },
  "quality_summary": {
    "GOOD": 115,
    "BLURRY": 5,
    "OVEREXPOSED": 0,
    "UNDEREXPOSED": 0,
    "LOW_FEATURE": 0,
    "CORRUPTED": 0
  },
  "observations": [ ... ]
}
```

### Observation Item Schema

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

---

## 3. Data Dictionary

### Visual Observation Fields

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `observation_id` | `string` | **Yes** | Stable, deterministic identifier (e.g. `frame_000001`) |
| `timestamp` | `float` | **Yes** | Monotonic capture timestamp in seconds ($t_k < t_{k+1}$) |
| `image` | `string` | **Yes** | Relative path to image file (`frames/frame_000001.jpg`) |
| `camera` | `object` | **Yes** | Camera geometry (`width`, `height`) and optional calibration (`intrinsics`, `distortion`) |
| `quality` | `object` | **Yes** | Quality condition report (`status`, `blur_score`, `quality_score`, `flags`) |
| `keyframe` | `boolean` | **Yes** | `true` if promoted to keyframe, `false` otherwise |

### Quality Status Enum

* `GOOD`: Sharp, well-exposed, feature-rich observation.
* `BLURRY`: Motion or defocus blur detected ($\sigma_{\Delta}^2 < \text{threshold}$).
* `OVEREXPOSED`: Luminance washed out ($\mu > 230$).
* `UNDEREXPOSED`: Insufficient illumination ($\mu < 30$).
* `LOW_FEATURE`: Low texture / feature count.
* `CORRUPTED`: Unreadable array or decoding failure.

### Camera Calibration Fields

* `width`, `height`: Image dimensions in pixels (always guaranteed).
* `intrinsics`: `fx`, `fy`, `cx`, `cy`, and $3\times 3$ `camera_matrix` (or `null` if uncalibrated).
* `distortion`: `coefficients` and `model` (or `null` if uncalibrated).
* `is_calibrated`: Explicit boolean flag.

---

## 4. Failure & Degradation Contract (Phase 11)

`S1Output` provides explicit top-level health indicators:

* **`status`**: `"completed"` (healthy), `"degraded"` (sparse/blurry frames), `"failed"` (unusable/corrupt).
* **`warnings`**: List of non-fatal operational warnings (e.g. `missing_camera_calibration`, `missing_uav_telemetry`, `insufficient_valid_observations`).
* **`errors`**: List of fatal error messages when status is `"failed"`.
* **`diagnostics`**: Structured dictionary containing observation counts, valid/corrupted breakdown, and sensor availability.

---

## 5. Guarantees & Preconditions

* **Non-Destructive Observations:** All candidate observations are preserved in `observations.json` with `keyframe: bool`. S2 can evaluate all observations or keyframes only.
* **Portable Relative Paths:** All `image` paths are relative to `s1_output/` root.
* **Stable IDs:** Observation identifiers remain immutable through S1 $\rightarrow$ S2 $\rightarrow$ S3.
* **Monotonic Timestamps:** Timestamps represent source video capture time in seconds, never wall-clock processing time.
* **Explicit Degradation:** Corrupt or invalid frames are never silently reported as valid.

---

## 6. Versioning

* **Contract Version:** 1.2.0
* **Schema Reference:** [`observations-schema.json`](observations-schema.json)