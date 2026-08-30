# S2 — Localization & Sensor Fusion

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

### Example pipelines:

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

**The absence of an optional sensor must not invalidate the S2 interface.**

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