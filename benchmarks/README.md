# AIRLOCK+ Benchmark Dataset

AIRLOCK+ is the primary benchmark dataset used to evaluate **Matrix**, the UAV-based one-pass 3D geospatial reconstruction pipeline.

The dataset contains aerial UAV video together with synchronized telemetry/GPS data. It is used to validate the Matrix pipeline progressively from **visual perception and localization through 3D reconstruction and georeferencing**.

## Dataset Structure

```text
benchmarks/dataset/
├── video-1005/
│   ├── telemetry.csv
│   └── video.mp4
├── video-1006/
│   ├── telemetry.csv
│   └── video.mp4
└── video-1007/
    ├── telemetry.csv
    └── video.mp4
```

Each flight directory is self-contained and contains:

* `video.mp4` — aerial UAV video used as the primary visual input.
* `telemetry.csv` — time-aligned UAV/vehicle positioning and motion data used as reference data during localization evaluation.

## Telemetry

The telemetry CSV contains one record per video frame and includes:

| Field                      | Description                                 |
| -------------------------- | ------------------------------------------- |
| `frame_index`              | Zero-based video frame index                |
| `video_time_s`             | Video-relative timestamp in seconds         |
| `timestamp`                | Absolute capture timestamp                  |
| `drone_lat`                | Drone latitude                              |
| `drone_lon`                | Drone longitude                             |
| `drone_altitude_m`         | Drone altitude in meters                    |
| `vehicle_lat`              | Vehicle latitude                            |
| `vehicle_lon`              | Vehicle longitude                           |
| `vehicle_altitude_m`       | Vehicle altitude in meters                  |
| `vehicle_altitude_wgs84_m` | WGS84-referenced vehicle altitude in meters |
| `vehicle_speed`            | Vehicle speed                               |
| `vehicle_data_status`      | Telemetry availability/status               |

Example:

```csv
frame_index,video_time_s,timestamp,drone_lat,drone_lon,drone_altitude_m,vehicle_lat,vehicle_lon,vehicle_altitude_m,vehicle_altitude_wgs84_m,vehicle_speed,vehicle_data_status
0,0.000000,2025-08-23 10:45:07.039000,30.289664,-97.782784,149.500000,30.289669,-97.782783,156.683101,131.952718,0.0,exact
```

## Role in Matrix

AIRLOCK+ is used to verify that each Matrix subsystem produces valid output for the next subsystem.

```text
AIRLOCK+ Video + Telemetry
          │
          ▼
     S1 — Visual
     Perception
          │
          ▼
     S1 Artifact
     Validation
          │
          ▼
     S2 — Localization
     & Sensor Fusion
          │
          ├──► Telemetry/GPS Evaluation
          │
          ▼
     S3 — 3D
     Reconstruction
          │
          ▼
     S4 — Georeferencing
     & Validation
          │
          ▼
     S5 — Application
     & Deployment
```

The benchmark is therefore not only an end-to-end dataset. It is also the basis for **progressive pipeline verification**.

## Evaluation Objectives

### S1 — Visual Perception

Verify that S1 correctly:

* extracts observations and frames;
* preserves stable observation identifiers;
* produces valid timestamps;
* identifies keyframes;
* reports camera information;
* reports image-quality information;
* produces a contract-compliant `observations.json`.

### S2 — Localization & Sensor Fusion

Evaluate:

* observation-to-pose correspondence;
* timestamp alignment;
* trajectory continuity;
* localization coverage;
* position accuracy against available telemetry;
* trajectory drift;
* handling of missing or degraded observations.

Telemetry should be treated as **reference positioning data**, not automatically as exact camera ground truth. The evaluation must account for differences between the recorded vehicle/drone position and the actual camera optical-center pose.

Where appropriate, WGS84 coordinates should be transformed into a local metric coordinate frame before calculating trajectory errors.

### S3 — 3D Reconstruction

Verify:

* valid point-cloud generation;
* finite 3D coordinates;
* valid point identifiers;
* valid mesh topology where applicable;
* reasonable reconstruction scale and spatial extent;
* sufficient reconstruction density;
* reconstruction quality metrics where available.

Visual inspection of the resulting reconstruction is also required.

### S4 — Georeferencing & Validation

Verify:

* valid coordinate reference information;
* correct transformation from the S3 local frame;
* plausible geographic placement;
* consistent scale and orientation;
* spatial validation metrics;
* explicit reporting of limitations and confidence.

## Flight Usage

The three selected flights are intended to provide progressively broader validation coverage.

| Flight       | Intended Role                      |
| ------------ | ---------------------------------- |
| `video-1005` | Development / reference flight     |
| `video-1006` | Validation / generalization flight |
| `video-1007` | Stress / robustness flight         |

These roles may be refined after inspecting the actual visual quality, trajectory characteristics, telemetry completeness, and scene complexity.

## Reproducibility

Benchmark data should be treated as **immutable input data**.

Pipeline code must not modify the original:

```text
video.mp4
telemetry.csv
```

Generated artifacts and evaluation results should be written outside the raw input files, for example:

```text
results/
└── airlock_plus/
    ├── video-1005/
    ├── video-1006/
    └── video-1007/
```

This allows the same benchmark inputs to be rerun against different Matrix implementations.

## Validation Philosophy

Matrix should not be considered validated merely because a subsystem executes without raising an exception.

Each stage must demonstrate that its **actual output artifact**:

1. satisfies the locked interface contract;
2. is internally consistent;
3. can be consumed by the next subsystem;
4. is numerically and physically plausible where applicable.

The primary engineering metric is therefore:

> **Longest verified pipeline**

rather than lines of code, number of implemented modules, or percentage of tasks completed.

For example:

```text
S1 ✓
  ↓
S2 ✓
  ↓
S3 ✓
  ↓
S4 ✗
```

means Matrix currently has a **verified S1→S3 pipeline**, not a completed S4 pipeline.

## Data Integrity

Do not:

* modify the original videos;
* modify the original telemetry;
* silently discard telemetry records;
* silently reinterpret timestamps;
* claim telemetry is camera ground truth without establishing the relationship;
* hardcode benchmark-specific assumptions into production subsystem code.

Any preprocessing or coordinate transformation required for evaluation should be explicit, reproducible, and documented.

## Future Extensions

The benchmark framework may later support:

* additional AIRLOCK+ flights;
* holdout evaluation flights;
* additional telemetry sources;
* camera/IMU metadata;
* automated trajectory metrics;
* reconstruction quality metrics;
* georeferencing accuracy metrics;
* automated regression testing;
* benchmark result tracking across Matrix versions.

The remaining AIRLOCK+ flights, where available, should preferably be retained as **unseen holdout data** rather than being used during development.
