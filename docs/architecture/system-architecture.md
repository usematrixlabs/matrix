# Matrix — System Architecture

This document describes the canonical system architecture, subsystem boundaries, data flow, interface contracts, and failure handling for **Matrix** (SIH26158).

---

## 1. System Overview

**Matrix** is a UAV video-to-3D geospatial reconstruction system that processes single-pass UAV video along with optional telemetry, GPS, IMU, and camera calibration into a georeferenced, validated 3D representation.

The five primary subsystems are:

| Subsystem | Name | Responsibility | Output Contract |
| :--- | :--- | :--- | :--- |
| **S1** | Visual Perception | Ingests video, extracts frames, assigns stable IDs, timestamps, quality scores, keyframe flags, evaluates degradation health, and packages `s1_output/` | `docs/architecture/contracts/perception-localization.md` |
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

## 3. Failure & Degradation Taxonomy (Phase 11)

Matrix recognizes five standardized status/error categories:

```text
Input Error  ──>  Processing Error  ──>  Quality Warning  ──>  Degraded Result  ──>  Completed
 (Hard Fail)        (Hard Fail)           (Non-blocking)         (Usable/Degraded)     (Healthy)
```

| Category | Trigger / Condition | System Behaviour | Pipeline Status |
| :--- | :--- | :--- | :--- |
| **Input Error** | Missing video, 0-byte file, corrupt header, unsupported codec | Halts pipeline immediately with explicit descriptive error | `"failed"` |
| **Processing Error** | Decoder crash, filesystem I/O error | Halts processing, logs diagnostic stack trace | `"failed"` |
| **Quality Warning** | Missing camera calibration, missing optional UAV telemetry | Continues processing with explicit `null` fields; appends warning | `"completed"` |
| **Degraded Result** | Insufficient visual observations ($< 5$ frames), high blur/featureless ratio ($> 80\%$) | Completes packaging but flags output for downstream caution | `"degraded"` |
| **Completed** | Valid stream, adequate observations, healthy quality metrics | Completes normal packaging | `"completed"` |

### S1 Diagnostics Object Schema

```json
{
  "health_status": "completed",
  "is_valid": true,
  "is_degraded": false,
  "degraded_reasons": [],
  "observations_summary": {
    "total_extracted": 120,
    "valid_count": 120,
    "corrupted_count": 0,
    "good_count": 115,
    "blurry_count": 5,
    "keyframes_selected": 24,
    "keyframe_density": 0.20
  },
  "sensor_availability": {
    "camera_calibration": true,
    "telemetry_present": false
  }
}
```

---

## 4. End-to-End Pipeline Sequence

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
                        │ • Health diagnostics      │
                        │ • Portable packaging      │
                        └─────────────┬─────────────┘
                                      │ s1_output/ (frames/ + observations.json)
                                      ▼
                        ┌───────────────────────────┐
                        │ S2 · Localization         │
                        │    & Sensor Fusion        │
                        │                           │
                        │ • Visual pose estimation  │
                        │ • EKF state filtering     │
                        │ • Telemetry fusion        │
                        │ • Trajectory smoothing    │
                        └─────────────┬─────────────┘
                                      │ S1 observations + Fused Poses
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

---

## 5. S2: Localization & Sensor Fusion

### Overview

S2 estimates camera poses, trajectories, and fusion by combining:

- **Visual localization:** PnP pose estimation from visual features (S1 observations)
- **Telemetry fusion:** Optional IMU, GPS, or other sensor data
- **State filtering:** Extended Kalman Filter (EKF) for robust, temporally consistent pose estimation

### State Representation

The S2 EKF maintains a 6-dimensional state vector:

```text
state = [x, y, z, vx, vy, vz]^T

where:
  x, y, z     = position (m) in local reconstruction coordinates
  vx, vy, vz  = velocity (m/s) in local reconstruction coordinates
```

### EKF Pipeline

1. **Prediction Step**
   - Time step: `dt` between successive observations
   - Motion model: constant velocity (kinematic model)
   - State propagation: `x_pred = F * x_prev` where `F` is the state transition matrix
   - Covariance update: `P_pred = F * P_prev * F^T + Q`

2. **Measurement Update Step**
   - Measurement: 3D visual position [x_visual, y_visual, z_visual] from pose estimation
   - Measurement matrix: `H` extracts position from state
   - Innovation (residual): `y = z_meas - H * x_pred`
   - Kalman gain: `K = P_pred * H^T * (H * P_pred * H^T + R)^-1`
   - State correction: `x_fused = x_pred + K * y`
   - Covariance update: `P_fused = (I - K * H) * P_pred`

3. **Dynamic Measurement Noise**
   - Measurement covariance `R` scales based on visual confidence score
   - High confidence → low measurement noise (trust visual estimate)
   - Low confidence → high measurement noise (trust prior state)

### Output Contract

S2 returns the original S2ObservationOutput with updated pose from the EKF filter:

```python
S2ObservationOutput(
    observation_id="frame_000123",
    timestamp=12.34,
    image="frames/frame_000123.jpg",
    pose=CameraPose(
        position=Position(x=fused_x, y=fused_y, z=fused_z),
        orientation=QuaternionOrientation(qx, qy, qz, qw)
    ),
    localization=LocalizationMeta(
        source=["visual", "fusion"],
        status="estimated",
        quality=LocalizationQuality(confidence=0.95)
    )
)
```

### Key Architectural Properties

1. **Coordinate System:** All poses are in **local reconstruction coordinates** (not geographic)
2. **Temporal Consistency:** EKF enforces smooth, physically plausible trajectories
3. **Confidence Propagation:** Measurement covariance adapts based on visual quality scores
4. **Observation Preservation:** All observations are fused; keyframe distinction is downstream
5. **State Accessibility:** State vector and covariance are internal; output is via updated observations

### Failure Conditions

- If no valid pose estimate can be obtained (e.g., degenerate point set), S2 logs a warning and may skip that observation
- If temporal data is missing, S2 assumes a default time step and logs a diagnostic
- If multiple observations arrive out-of-order, S2 resets its temporal state to maintain monotonicity

---

## 6. S2 → S3 Interface

See [S2 → S3 Interface Contract](contracts/localization-reconstruction.md).

---

## 7. S3: 3D Reconstruction

### Overview

S3 generates the 3D dense/sparse representation of the observed scene from multi-view feature tracks and localized camera poses.

### Key Architectural Properties

1. **Multi-View Triangulation:** Linear DLT / SVD triangulation with cheirality checks.
2. **Quality Evaluation:** Mean and median reprojection error computation with statistical outlier filtering.
3. **Local Spatial Frame:** Coordinates remain in `S3_LOCAL` meters with bounding box metadata.
4. **Standard Artifacts:** Outputs standard binary/ASCII `scene.ply` and structured `metadata.json`.

---

## 8. S3 → S4 Interface

See [S3 → S4 Interface Contract](contracts/reconstruction-georeferencing.md).

S3 provides the local 3D reconstruction (`PointCloudData` / `ReconstructionInput`), color arrays, and reconstruction quality metadata to S4.

---

## 9. S4: Georeferencing & Validation

### Overview

S4 transforms the local 3D reconstruction into real-world geographic coordinates (e.g. WGS 84 / UTM) and evaluates spatial accuracy.

### Core Capabilities

1. **7-Parameter Similarity (Helmert) Transformation:** Scale, rotation matrix, and translation vector estimation.
2. **MSAC Robust Outlier Rejection:** Discards erroneous GCP correspondences.
3. **Horizontal & Vertical Error Split:** Computes independent $\text{RMSE}_{\text{3D}}$, $\text{RMSE}_{\text{Horizontal}}$, and $\text{RMSE}_{\text{Vertical}}$ with tolerance checks.
4. **Spatial Consistency Analysis:** $k$-NN neighbor distances, terrain plane fit residual RMSE, and relative scale preservation.
5. **Quality & Limitations Detection:** Auto-detects GCP geometry caveats and assigns confidence levels.

---

## 10. S4 → S5 Interface

See [S4 → S5 Interface Contract](contracts/georeferencing-application.md).

S4 delivers the georeferenced 3D scene, validation metrics, CRS metadata, quality status, and known limitations to S5.

---

## 11. S5: Application & Deployment

### Overview

S5 is the system-facing orchestration layer. It manages the complete end-to-end execution lifecycle ($S1 \to S2 \to S3 \to S4 \to S5$), job dispatch, deliverables packaging, and user interaction.

### Core Capabilities

1. **Pipeline Orchestrator:** `Orchestrator.run_pipeline()` coordinates execution from raw UAV video or intermediate representations.
2. **Deliverables Packaging:** Assembles deliverables (`scene.ply`, `s2_output.json`, `georeferencing_report.html`, `pipeline_manifest.json`).
3. **Stage Metrics & Telemetry:** Monitors per-stage execution times, status, points reconstructed, and accuracy metrics.
