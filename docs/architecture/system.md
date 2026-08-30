# Matrix — System Architecture

## 1. System Overview

**Matrix** is a UAV-based video-to-3D geospatial reconstruction system designed to transform a **single-pass UAV video and any available flight, location, camera, or sensor information** into a usable, georeferenced 3D representation of an observed environment.

The system follows a **video-first architecture**.

The minimum supported input is:

> **UAV video alone.**

When additional information such as GPS, GNSS, IMU, altitude, camera calibration, RTK/PPK, flight telemetry, or other metadata is available, Matrix may use it to improve localization and downstream reconstruction.

The system is composed of five coordinated subsystems:

1. **S1 — Visual Perception**
2. **S2 — Localization & Sensor Fusion**
3. **S3 — 3D Reconstruction**
4. **S4 — Georeferencing & Validation**
5. **S5 — Application & Deployment**

The architecture is modular: each subsystem owns its internal implementation while communicating with other subsystems through explicit contracts.

---

# 2. Subsystem Responsibilities

The fundamental responsibility boundaries are:

| Subsystem                             | Owns                                                                                                                                                  | Does Not Own                                                                                 |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| **S1 — Visual Perception**            | UAV input ingestion, video processing, visual observation preparation, and preservation/forwarding of available input metadata and sensor information | Localization, sensor fusion, camera trajectory estimation, 3D reconstruction, georeferencing |
| **S2 — Localization & Sensor Fusion** | Camera localization, trajectory, pose estimation, and interpretation/fusion of available visual and sensor/location information                       | Primary video processing, 3D reconstruction, final georeferencing, application               |
| **S3 — 3D Reconstruction**            | Generation of the 3D representation from visual observations and camera information                                                                   | Primary localization, sensor fusion, final geographic alignment, application                 |
| **S4 — Georeferencing & Validation**  | Geographic alignment and validation of the reconstructed scene                                                                                        | Primary reconstruction, localization, application                                            |
| **S5 — Application & Deployment**     | Pipeline orchestration, user interaction, visualization, and deployment                                                                               | Internal algorithms of S1–S4                                                                 |

The core mental model is:

> **S1 answers: What did we observe?**
>
> **S2 answers: Where was the camera and how did it move?**
>
> **S3 answers: What does the observed environment look like in 3D?**
>
> **S4 answers: Where is that 3D scene in the real world, and how accurate is it?**
>
> **S5 answers: How does the user run and interact with Matrix?**

---

# 3. S1 — Visual Perception

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

S1 is also responsible for **carrying available UAV-side information through the S1 → S2 boundary**.

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

### S1 does not interpret this information as a localization solution.

For example:

> S1 may provide GPS coordinates and IMU measurements to S2.

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

---

# 4. S1 → S2 Interface

S1 provides S2 with the information required for localization.

### S1 → S2 contains

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

Therefore:

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

Likewise:

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

### Architectural Rule

> **S1 transports and associates available input information. S2 interprets and uses that information for localization and sensor fusion.**

S1 must not silently discard relevant sensor/location information supplied with the UAV input.

S2 must not assume that any particular optional sensor is always present.

---

# 5. S2 — Localization & Sensor Fusion

## Responsibility

S2 determines **camera movement, trajectory, position, and pose** using the visual observations and any available sensor/location information supplied by S1.

S2 may perform:

* Visual localization
* Camera motion estimation
* Camera pose estimation
* Trajectory estimation
* GPS/GNSS integration
* IMU integration
* Sensor fusion
* Position estimation
* Orientation estimation
* Coordinate/reference handling required for localization
* Localization quality/confidence estimation
* Association between observations and poses

S2 may use any subset of the available information.

For example:

```text
Video only
    → Visual localization

Video + GPS
    → Visual + GPS localization

Video + GPS + IMU
    → Visual + GPS + IMU fusion

Video + RTK/PPK
    → Localization using high-accuracy position information
```

The absence of an optional sensor must not invalidate the S2 interface.

### S2 does not own

* Primary video processing
* Frame extraction
* Final 3D reconstruction
* Point-cloud generation
* Mesh generation
* Final geographic alignment
* Final spatial validation
* User interface
* Application orchestration

### S2 Boundary

```text
             S1 OUTPUT
                 │
        ┌────────┴────────┐
        │                 │
Visual observations   Sensor/location
                     information
        │                 │
        └────────┬────────┘
                 ▼
                S2
                 │
                 ▼
       CAMERA LOCALIZATION
       + TRAJECTORY
       + POSE
       + QUALITY
```

### S2 answers:

> **Where was the camera, how did it move, and how confident are we in that estimate?**

---

# 6. S2 → S3 Interface

S2 provides S3 with **both the visual observations originally prepared by S1 and the localization information generated by S2**.

This is a deliberate architectural decision.

### S2 → S3 contains

```text
S2 OUTPUT
│
├── S1 VISUAL DATA
│   ├── Frames / keyframes
│   ├── Observation identifiers
│   ├── Timestamps
│   └── Visual metadata
│
└── S2 LOCALIZATION DATA
    ├── Camera poses
    ├── Camera trajectory
    ├── Position information
    ├── Orientation information
    ├── Coordinate/reference information
    └── Localization quality/status
```

### Why S2 carries S1 data forward

S3 needs to know both:

> **What was observed?**

and:

> **Where and when was it observed?**

S1 provides the observations.

S2 associates those observations with camera movement and pose.

Therefore:

```text
                    S1
                     │
             Visual observations
                     │
                     ▼
                    S2
                     │
           ┌─────────┴─────────┐
           │                   │
      S1 visual data     S2 localization
           │                   │
           └─────────┬─────────┘
                     ▼
                    S3
```

S3 should not need to independently retrieve S1 output.

### S2 → S3 Association Guarantee

For each localization result, S2 must provide a documented association to the relevant observation through identifiers and, where available, timestamps.

Conceptually:

```text
Observation ID
      │
      ├── Frame / image
      ├── Timestamp
      └── Camera pose
```

This allows S3 to determine which camera pose corresponds to which visual observation.

---

# 7. S3 — 3D Reconstruction

## Responsibility

S3 generates the **3D representation of the observed environment** using the visual observations and camera information provided by S2.

S3 may perform:

* Multi-view reconstruction
* Structure-from-motion-related processing
* Depth estimation
* Point-cloud generation
* Surface reconstruction
* Mesh generation
* Texture generation where applicable
* Reconstruction quality assessment

### S3 does not own

* Primary video ingestion
* Primary frame extraction
* Primary localization
* GPS/IMU fusion
* Final geographic alignment
* Final geographic validation
* User interface
* Application orchestration

Internal reconstruction methods may perform local optimization or refinement where required. Such implementation details do not change the subsystem boundary.

### S3 answers:

> **What does the observed environment look like in 3D?**

---

# 8. S3 → S4 Interface

S3 provides:

* 3D reconstruction
* Point cloud and/or mesh
* Reconstruction metadata
* Relevant camera/trajectory information
* Spatial/reference metadata
* Reconstruction quality information

Conceptually:

```text
S3
 │
 ▼
3D RECONSTRUCTION
 │
 ├── Point cloud
 ├── Mesh
 ├── Metadata
 ├── Spatial reference
 └── Quality information
 │
 ▼
S4
```

S3 produces a valid representation according to the agreed reconstruction coordinate/reference convention.

S3 does not assume that its local reconstruction coordinates are geographic coordinates.

---

# 9. S4 — Georeferencing & Validation

## Responsibility

S4 transforms the reconstruction into a **geographically meaningful representation** and evaluates its spatial quality.

S4 owns:

* Geographic alignment
* Coordinate transformations
* Reference-system handling
* Georeferencing
* Spatial consistency checks
* Geographic accuracy evaluation
* Validation metrics
* Quality/confidence reporting
* Known limitations

### S4 does not own

* Video processing
* Primary localization
* Sensor fusion
* Primary 3D reconstruction
* Mesh generation
* Application UI
* Pipeline orchestration

### S4 answers:

> **Where is the reconstructed scene in the real world, and how accurate is it?**

---

# 10. S4 → S5 Interface

S4 provides:

* Georeferenced 3D scene
* Geographic metadata
* Coordinate reference information
* Validation metrics
* Quality/confidence information
* Visualization metadata
* Known limitations/status

```text
S4
 │
 ▼
GEOREFERENCED + VALIDATED SCENE
 │
 ├── 3D data
 ├── Geographic reference
 ├── Metrics
 └── Quality/status
 │
 ▼
S5
```

S5 should be able to present the result without needing to understand the internal algorithms used by S4.

---

# 11. S5 — Application & Deployment

## Responsibility

S5 is the **system-facing layer**.

S5 owns:

* User input
* File upload
* Pipeline initiation
* Pipeline orchestration
* Job/process management
* Processing status
* Error/status presentation
* Result delivery
* 3D visualization
* Application deployment
* Runtime integration

S5 connects the processing pipeline to the user.

### S5 does not own

* Visual perception algorithms
* Localization algorithms
* Sensor-fusion algorithms
* Reconstruction algorithms
* Georeferencing algorithms
* Validation methodology

S5 **orchestrates** these capabilities; it does not absorb their responsibilities.

### S5 answers:

> **How does a user run Matrix and interact with its output?**

---

# 12. End-to-End Data Flow

The canonical Matrix data flow is therefore:

```text
                         UAV INPUT
                             │
             ┌───────────────┴───────────────┐
             │                               │
           VIDEO                     GPS / GNSS / IMU
             │                       RTK / PPK / etc.
             │                               │
             └───────────────┬───────────────┘
                             ▼
                    ┌─────────────────┐
                    │ S1 · PERCEPTION │
                    │                 │
                    │ Visual          │
                    │ observations    │
                    │ + available     │
                    │ input data      │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ S2 · LOCALIZATION│
                    │    & SENSOR      │
                    │    FUSION        │
                    │                 │
                    │ S1 observations │
                    │ + sensors       │
                    │ + localization   │
                    └────────┬────────┘
                             │
                             │ Combined S1 + S2
                             ▼
                    ┌─────────────────┐
                    │ S3 · 3D         │
                    │    RECONSTRUCTION│
                    │                 │
                    │ Point cloud     │
                    │ Mesh / scene    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ S4 · GEO +      │
                    │    VALIDATION   │
                    │                 │
                    │ Geographic      │
                    │ alignment +     │
                    │ validation      │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ S5 · APPLICATION│
                    │    + DEPLOYMENT │
                    └────────┬────────┘
                             │
                             ▼
                       3D GEO OUTPUT
```

The critical architectural relationship is:

```text
S1
 │
 │ Visual observations
 │ + available UAV input data
 ▼
S2
 │
 │ S1 data
 │ + localization
 ▼
S3
 │
 │ 3D reconstruction
 ▼
S4
 │
 │ Georeferenced + validated scene
 ▼
S5
```

---

# 13. Core Architectural Rules

### Rule 1 — S1 Does Not Own Localization

S1 may ingest, extract, preserve, and forward GPS/GNSS/IMU and other sensor information.

**S2 owns interpretation and fusion.**

### Rule 2 — S2 Does Not Own Reconstruction

S2 provides camera position, trajectory, pose, and associated quality information.

**S3 owns reconstruction.**

### Rule 3 — S2 Carries S1 Data Forward

S3 receives the visual observations originally prepared by S1 together with S2's localization information.

### Rule 4 — S3 Does Not Own Final Georeferencing

S3 may produce a reconstruction in a local coordinate system.

**S4 owns final geographic alignment and spatial validation.**

### Rule 5 — S5 Does Not Own Processing Intelligence

S5 orchestrates and exposes S1–S4.

It does not redefine their internal algorithms.

### Rule 6 — Optional Data Remains Optional

GPS, GNSS, IMU, RTK/PPK, altitude, and other sensor information may improve the system but must not silently become mandatory dependencies.

### Rule 7 — No Hidden Dependencies

A subsystem may only depend on information explicitly defined by its interface.

### Rule 8 — Preserve Observation Association

Visual observations, timestamps, identifiers, sensor measurements, and camera poses must remain associable throughout the pipeline.

### Rule 9 — Validate at Boundaries

Consumers validate incoming data rather than blindly assuming upstream output is valid.

### Rule 10 — One Owner for Each Responsibility

Each capability has one primary subsystem owner.

---

# 14. One-Line Responsibility Map

```text
S1 → Observe
S2 → Localize
S3 → Reconstruct
S4 → Georeference & Validate
S5 → Orchestrate & Present
```

Or, more explicitly:

> **S1 tells us what we saw.**
>
> **S2 tells us where/how the camera moved.**
>
> **S3 turns those observations into 3D.**
>
> **S4 puts that 3D scene into geographic context and validates it.**
>
> **S5 turns the pipeline into a usable product.**

This is the canonical responsibility boundary for Matrix.
