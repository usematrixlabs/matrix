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
| **Orchestrator** | Pipeline (`src/pipeline/`) | Owns composition — invokes S1–S5, adapts between contracts, manages artifacts, propagates status | See [§5 Pipeline Orchestrator](#5-pipeline-orchestrator-srcpipeline) and [§6 ADR-002](#6-independent-subsystems--pipeline-owned-integration-adr-002) |

> **Adopted principle:** *Subsystems own computation. The pipeline owns composition. Contracts own boundaries.* — Full rationale in [ADR-002](../../decisions/ADR-002-independent-subsystems-pipeline-owned-integration.md) and `docs/architecture/system.md` §15.

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

## 5. Pipeline Orchestrator (`src/pipeline/`)

The orchestrator is a thin wrapper that invokes each subsystem's main
exported entry point in sequence and passes outputs from one stage to
the next. It does **not** implement any S1–S4 algorithms itself and does
not reach into any subsystem's internals — every cross-subsystem
boundary it touches is one of the documented interface contracts in
`docs/architecture/contracts/`.

### 5.1 Responsibility

* Create a per-run output directory.
* Invoke S1, S2, S3, S4, S5 in order.
* Pass stage outputs forward through their documented contracts.
* Preserve all outputs and stop clearly on failure.
* Emit a `PipelineResult` describing success / failure and the path of
  the final bundled output.

### 5.2 Entry Point

```python
from src.pipeline.orchestrator import run_pipeline

result = run_pipeline(
    video_path="benchmarks/dataset/video-1005/video.mp4",
    gps_path="benchmarks/dataset/video-1005/gps.csv",
    output_dir="benchmarks/results/video-1005",
)
```

CLI form:

```bash
python -m src.pipeline.orchestrator \
    --video benchmarks/dataset/video-1005/video.mp4 \
    --gps   benchmarks/dataset/video-1005/gps.csv \
    --output benchmarks/results/video-1005
```

### 5.3 Stage Wiring

| Stage | Subsystem | Public Entry Point                          | Input Contract                                        | Output Contract                                  |
| :--- | :--- | :------------------------------------------ | :---------------------------------------------------- | :----------------------------------------------- |
| S1 | Visual Perception | `src.visual_perception.run_s1` | `--video` path | `S1Output` → `S1Contract` (`s1/observations.json` + `s1/frames/`) |
| S2 | Localization & Sensor Fusion | `src.localization_sensor_fusion.run_s2` | `S1Contract` + `--gps` CSV | `S2Contract` (`s2/s2_output.json`) |
| S3 | 3D Reconstruction | `src.reconstruction.run_s3` | `S2Contract` + `image_root` | `S3Contract` (`s3/scene.ply` + `s3/metadata.json`) |
| S4 | Georeferencing & Validation | `src.georeferencing_validation.run_s4` | `S3Contract` | `S4Contract` (`s4/georeferenced.ply` + `s4/georeferencing.json`) |
| S5 | Application & Deployment | `src.application_deployment.run_s5` | `S4Contract` + per-stage metadata | `S5Contract` (`s5/final_output.json`) |

The GPS CSV enters the system through **S2** (sensor fusion), not
through the orchestrator as a separate processing stage.

### 5.4 Output Layout

For one run, the orchestrator creates the following layout under the
user-supplied output directory. Each subsystem owns its own subdirectory
and the orchestrator never reads another subsystem's internal files
beyond the documented contract outputs.

```text
output_dir/
├── s1/
│   ├── observations.json
│   └── frames/
├── s2/
│   └── s2_output.json
├── s3/
│   ├── scene.ply
│   └── metadata.json
├── s4/
│   ├── georeferenced.ply
│   └── georeferencing.json
└── s5/
    └── final_output.json
```

### 5.5 Failure Handling

The orchestrator catches all stage exceptions at the pipeline boundary,
records the failing stage in `PipelineResult.failed_stage`, prints a
stack trace to stderr, and returns a `PipelineResult(success=False, ...)`.
All artifacts produced by stages that ran successfully are preserved on
disk so they can be inspected or replayed.

Stages are also allowed to record **degraded** outcomes (e.g., S4 with
an empty input point cloud) rather than failing the whole pipeline.
These are surfaced through `stage_status` and the `sN/` summary files.
---

## 8. S3: 3D Reconstruction

### Overview

S3 generates the 3D dense/sparse representation of the observed scene from multi-view feature tracks and localized camera poses.

### Key Architectural Properties

1. **Multi-View Triangulation:** Linear DLT / SVD triangulation with cheirality checks.
2. **Quality Evaluation:** Mean and median reprojection error computation with statistical outlier filtering.
3. **Local Spatial Frame:** Coordinates remain in `S3_LOCAL` meters with bounding box metadata.
4. **Standard Artifacts:** Outputs standard binary/ASCII `scene.ply` and structured `metadata.json`.

---

## 9. S3 → S4 Interface

See [S3 → S4 Interface Contract](contracts/reconstruction-georeferencing.md).

S3 provides the local 3D reconstruction (`PointCloudData` / `ReconstructionInput`), color arrays, and reconstruction quality metadata to S4.

---

## 10. S4: Georeferencing & Validation

### Overview

S4 transforms the local 3D reconstruction into real-world geographic coordinates (e.g. WGS 84 / UTM) and evaluates spatial accuracy.

### Core Capabilities

1. **7-Parameter Similarity (Helmert) Transformation:** Scale, rotation matrix, and translation vector estimation.
2. **MSAC Robust Outlier Rejection:** Discards erroneous GCP correspondences.
3. **Horizontal & Vertical Error Split:** Computes independent $\text{RMSE}_{\text{3D}}$, $\text{RMSE}_{\text{Horizontal}}$, and $\text{RMSE}_{\text{Vertical}}$ with tolerance checks.
4. **Spatial Consistency Analysis:** $k$-NN neighbor distances, terrain plane fit residual RMSE, and relative scale preservation.
5. **Quality & Limitations Detection:** Auto-detects GCP geometry caveats and assigns confidence levels.

---

## 11. S4 → S5 Interface

See [S4 → S5 Interface Contract](contracts/georeferencing-application.md).

S4 delivers the georeferenced 3D scene, validation metrics, CRS metadata, quality status, and known limitations to S5.

---

## 12. S5: Application & Deployment

### Overview

S5 is the system-facing orchestration layer. It manages the complete end-to-end execution lifecycle ($S1 \to S2 \to S3 \to S4 \to S5$), job dispatch, deliverables packaging, and user interaction.

### Core Capabilities

1. **Pipeline Orchestrator:** `src.pipeline.run_pipeline()` (the sole integration owner) drives S1 → S5. S5 itself exposes only :func:`run_s5`; cross-subsystem orchestration lives in `src/pipeline/`.
2. **Deliverables Packaging:** Assembles deliverables (`scene.ply`, `s2_output.json`, `georeferencing_report.html`, `pipeline_manifest.json`).
3. **Stage Metrics & Telemetry:** Monitors per-stage execution times, status, points reconstructed, and accuracy metrics.

### 5.6 Pipeline as Integration Owner (ADR-002)

Per [ADR-002](../../decisions/ADR-002-independent-subsystems-pipeline-owned-integration.md), the pipeline is the **sole integration owner**. Subsystems do not read each other's private artifacts or internal code; the pipeline obtains upstream outputs, adapts between documented contracts in `docs/architecture/contracts/`, and passes validated inputs downstream. This prevents hidden coupling and makes integration failures attributable to the pipeline adapter rather than to downstream algorithms.

---

## 6. Subsystem Public Surface

Each subsystem is a **sealed module**. It exposes exactly one integration
symbol (a `run_sN` function) plus the canonical Pydantic contract type
it produces. Anything else lives under a private `_internal/` namespace
and is **not** importable from outside the subsystem package.

### 6.1 Public Symbols

| Subsystem | Integration Symbol | Contract Type |
| :--- | :--- | :--- |
| **S1** — Visual Perception | `run_s1(video_path, output_dir, config=None) -> S1Output` | `S1Contract` / `S1Output.to_contract() -> S1Contract` |
| **S2** — Localization & Sensor Fusion | `run_s2(s1_contract, gps_path, output_dir, config=None) -> S2Contract` | `S2Contract` (matches `s2_output.json` schema) |
| **S3** — 3D Reconstruction | `run_s3(s2_contract, image_root, output_dir, config=None) -> S3Contract` | `S3Contract` (`point_cloud`, `metadata`, `spatial_reference`) |
| **S4** — Georeferencing & Validation | `run_s4(s3_contract, output_dir, config=None) -> S4Contract` | `S4Contract` (`georeferenced_scene`, `validation_metrics`, `coordinate_reference`) |
| **S5** — Application & Deployment | `run_s5(s4_contract, output_dir, success, stage_status, artifacts, summary, config=None) -> S5Contract` | `S5Contract` (`manifest`, `artifacts`, `summary`) |

### 6.2 Directory Layout

Every subsystem follows the same physical structure:

```text
src/<subsystem>/
├── __init__.py        # public API only — single import surface
├── interface.py       # the single runner (run_sN)
└── _internal/         # everything else, leading underscore
    ├── contracts.py   # producer-owned Pydantic boundary types
    └── ...            # algorithms, adapters, helpers
```

The pipeline (`src/pipeline/`) imports **only** each subsystem's public
surface (`src.<subsystem>`). Subsystems never import each other.

### 6.3 Isolation Rule

The canonical principle is unchanged:

> **Subsystems own computation. The pipeline owns composition. Contracts own boundaries.**

In code, this translates to:

```python
# ALLOWED — orchestrator imports the public surface:
from src.visual_perception import run_s1, S1Output
from src.localization_sensor_fusion import run_s2, S2Contract
from src.reconstruction import run_s3, S3Contract
from src.georeferencing_validation import run_s4, S4Contract
from src.application_deployment import run_s5, S5Contract

# FORBIDDEN — any of these is a CI violation:
from src.visual_perception.frame_extractor import FrameExtractor
from src.localization_sensor_fusion._internal.adapters.s1_adapter import S1InputAdapter
from src.reconstruction.models.schema import S2Payload        # S3 reaching into itself is fine,
                                                              # but S2 / S4 must not reach here.
```

The forbidden imports are enforced by `tests/test_isolation.py`:

* No module under `src/<subsystem>/` may `import` a sibling subsystem
  package (`src.<other_subsystem>`).
* `src/<subsystem>/interface.py` may import nothing outside its own
  `_internal/` namespace.
* Outside `src/pipeline/`, no module may import across subsystem
  boundaries. The orchestrator is the single allowed cross-subsystem
  importer.

### 6.4 Contract Ownership

Each producer owns its contract type under its own `_internal/contracts.py`:

| Producer | Contract Type | Boundary |
| :--- | :--- | :--- |
| S1 | `S1Contract` | `docs/architecture/contracts/perception-localization.md` |
| S2 | `S2Contract` | `docs/architecture/contracts/localization-reconstruction.md` |
| S3 | `S3Contract` | `docs/architecture/contracts/reconstruction-georeferencing.md` |
| S4 | `S4Contract` | `docs/architecture/contracts/georeferencing-application.md` |
| S5 | `S5Contract` | internal S4 → S5 wire format |

The orchestrator is the **only** place that constructs an `S2Contract`
from an `S1Contract`, an `S3Contract` from an `S2Contract`, etc.
Subsystems no longer accept dicts shaped like other subsystems'
outputs — they receive their upstream wire-format contract and the
producer's documented fields are accessed via duck-typed attribute
access (S3 reading S2's `S2Contract.observations`, etc.).

### 6.5 Why This Matters

Sealed subsystems and the CI guard deliver three guarantees that the
older "everything is public" layout could not:

1. **A subsystem can be swapped, refactored, or rewritten as long as
   its `run_sN` and `S<N>Contract` honor the contract.** Internal
   modules in `_internal/` are free to change without coordinating with
   other subsystem owners.
2. **Integration failures are attributable.** When a downstream stage
   crashes, the only places a violation could have been introduced are
   the producer's contract output, the orchestrator's adaptation, or
   the consumer's input validation — not a third subsystem quietly
   reaching into a sibling.
3. **The contract types become the currency.** Once `S1Contract` etc.
   are Pydantic-validated wire formats, the orchestrator can stop
   carrying ad-hoc dicts around; contracts become the canonical data
   flowing through `src/pipeline/`.

---

## 7. Independent Subsystems & Pipeline-Owned Integration (ADR-002)

**Status: Adopted** — See [ADR-002](../../decisions/ADR-002-independent-subsystems-pipeline-owned-integration.md) and `system.md` §15.

> **Subsystems own computation. The pipeline owns composition. Contracts own boundaries.**

### 7.1 Responsibility Boundary

```text
┌─────────────────┐
│ S1              │
│ Visual          │
│ Perception      │
└────────┬────────┘
         │
         │ S1 contract
         ▼
┌─────────────────────────────────────┐
│             PIPELINE                │
│  • orchestration • adaptation       │
│  • data movement • execution order  │
│  • artifact management              │
│  • status propagation               │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────┐
│ S2              │
│ Localization &  │
│ Sensor Fusion   │
└────────┬────────┘
         │ S2 contract
         ▼
       PIPELINE → S3 → PIPELINE → S4 → PIPELINE → S5
```

* **Subsystem** owns: algorithms, contract compliance, input validation, outputs/metrics/diagnostics/status, degraded-input handling.
* **Pipeline** (`src/pipeline/`) owns: execution order, invocation, obtaining/adapting/passing data between contracts, artifact locations, status/failure propagation, orchestration, cross-subsystem validation. It is not part of any subsystem's internals.

### 7.2 Data Ownership — Pipeline-Mediated Only

```text
S1 output → Pipeline obtains → Pipeline adapts → S2 input
S2 output → Pipeline obtains → Pipeline adapts → S3 input
S3 output → Pipeline obtains → Pipeline adapts → S4 input
S4 output → Pipeline obtains → Pipeline adapts → S5 input
```

No subsystem reads another subsystem's private artifacts directly. Adapters live in `src/pipeline/`.

### 7.3 What a Subsystem Must NOT Do

* Reach into another subsystem's internal code or private artifacts.
* Assume how upstream data was generated.
* Perform another subsystem's algorithmic responsibilities.
* Implement pipeline orchestration.
* Make assumptions about input source beyond its contract.

Example: S3 does not care whether camera poses came from GPS, visual odometry, EKF, COLMAP, or synthetic data — only that the S3 input contract is satisfied.

### 7.4 Contract-First Integration

Every boundary in `docs/architecture/contracts/` must define input (required/optional fields, types, units, coordinate frames, valid ranges, quality requirements) and output (artifacts, schemas, metrics, status, diagnostics, failure/degradation semantics). The pipeline converts one contract's output into the next contract's input.

### 7.5 Status Classification

Three statuses must not be conflated:

| # | Status | Question |
|---|--------|----------|
| 1 | **Module status** | Is the subsystem itself implemented correctly? |
| 2 | **Integration status** | Is the pipeline correctly connecting the subsystem? |
| 3 | **End-to-end status** | Does the complete system process real benchmark data? |

Example:

```text
S3 module:             ✅ Implemented
S2 → S3 integration:   ❌ Invalid/incomplete
End-to-end pipeline:   ❌ Fails
```

A zero-point S3 result does not automatically imply an S3 defect if the pipeline failed to provide valid camera geometry — that is a pipeline integration issue if S3 correctly handled invalid input per its contract.

### 7.6 Engineering Rule — Trace to the Contract Boundary

```text
Did upstream produce valid output? ──NO→ upstream issue
        │ YES
Did pipeline correctly adapt/pass it? ──NO→ pipeline integration issue
        │ YES
Did downstream correctly process it? ──NO→ downstream issue
        │ YES → investigate subsequent boundary
```

### 7.7 Current Assessment (under this principle)

| Component | Assessment |
|-----------|------------|
| **S1** | Mostly implemented; calibration strategy for uncalibrated inputs remains |
| **S2** | Core components exist; visual localization execution/fusion remains incomplete |
| **S3** | Core reconstruction substantially complete; real-data validation & interface formalization remain |
| **S4** | Core georeferencing exists; real control-point/CRS/Helmert workflow remains |
| **S5** | Largely incomplete; primarily a bundling stub |
| **Pipeline** | Significant integration work remains (adaptation, S2 invocation, status propagation, multi-flight) |
