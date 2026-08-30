# S1 — Visual Perception

## Responsibility

S1 transforms the raw UAV input into **usable, ordered visual observations** and preserves input information required by downstream subsystems.

S1 may perform:

* Video ingestion
* Video decoding
* Frame extraction
* Frame selection/keyframing
* Image preprocessing
* Frame/observation identification
* Temporal ordering
* Timestamp association
* Visual quality assessment
* Visual observation generation
* Extraction of available UAV metadata

### S1 carries available UAV-side information through the S1 → S2 boundary

This may include:

* GPS
* GNSS
* IMU
* Altitude
* RTK/PPK information
* Camera metadata
* Flight telemetry
* Other available sensor information
* Relevant timestamps and identifiers

**S1 does not interpret this information as a localization solution.**

For example:

> S1 may provide GPS coordinates and IMU measurements to S2.
>
> S2 determines how those measurements should be interpreted, fused, and used for localization.

### S1 Boundary

```text
                    UAV INPUT
                        │
          ┌─────────────┴─────────────┐
          │                           │
        VIDEO                 GPS / GNSS / IMU
          │                    / TELEMETRY /
          │                   CAMERA METADATA
          └─────────────┬─────────────┘
                        ▼
                       S1
                        │
                        ▼
              VISUAL OBSERVATIONS
                        +
              AVAILABLE INPUT DATA
```

### S1 answers:

> **What did we observe, and what information did the UAV provide about that observation?**

## Interface

### S1 → S2 Output

```text
S1 OUTPUT
│
├── Visual observations
│   ├── Frames / keyframes
│   ├── Observation identifiers
│   ├── Frame ordering
│   └── Visual metadata
│
├── Temporal information
│   └── Timestamps where available
│
└── Available UAV information
    ├── GPS
    ├── GNSS
    ├── IMU
    ├── Altitude
    ├── RTK/PPK
    ├── Camera metadata
    ├── Flight telemetry
    └── Other available sensor information
```

The availability of these additional inputs is **optional**.

### Valid pipelines:

```text
VIDEO ONLY
    │
    ▼
   S1
    │
    ▼
   S2
```

is valid.

```text
VIDEO + GPS + IMU + GNSS + TELEMETRY
                │
                ▼
               S1
                │
                ▼
               S2
```

is valid.