# S2 → S3 Interface Contract

## Overview

S2 (Localization & Sensor Fusion) provides both the original visual observations from S1 and the camera localization data (poses, trajectory) to S3 (3D Reconstruction) for multi-view 3D reconstruction.

## Inputs

### S1 Visual Data (forwarded from S1)

| Field | Type | Description |
| --- | --- | --- |
| `frames` | `array<Frame>` | Ordered list of image frames / keyframes |
| `keyframes` | `array<Keyframe>` | Selected keyframes for reconstruction |
| `observation_identifiers` | `array<string>` | Unique identifiers for each observation |
| `timestamps` | `array<float>` | Capture timestamps corresponding to observations |
| `visual_metadata` | `object` | Per-frame quality and feature metadata |

### S2 Localization Data

| Field | Type | Description |
| --- | --- | --- |
| `camera_poses` | `array<Pose>` | Camera pose (position + orientation) per frame |
| `camera_trajectory` | `array<Pose>` | Sequence of camera positions over time |
| `position_information` | `object` | Estimated camera positions with confidence |
| `orientation_information` | `object` | Estimated camera orientations with confidence |
| `coordinate_reference` | `object` | Reference frame/coordinate system used for poses |
| `localization_quality` | `object` | Confidence scores, status flags per pose |
| `pose_timestamps` | `array<float>` | Timestamps corresponding to each pose |

### Pose (individual)

| Field | Type | Description |
| --- | --- | --- |
| `position` | `array<float>[3]` | X, Y, Z coordinates in local reconstruction coordinate system |
| `orientation` | `array<float>[3]` or `quaternion` | Roll, pitch, yaw or quaternion representation |
| `pose_id` | `string` | Unique identifier for this pose |
| `timestamp` | `float` | Time at which this pose was estimated |
| `covariance` | `array<float>[6]` | uncertainty estimates (if available) |

## Outputs

S3 generates a 3D reconstruction including point cloud, mesh, and associated metadata.

## Preconditions

- Each S2 localization result must be associable to an S1 observation via `frame_id`, `timestamp`, or index
- The coordinate reference system used by S2 must be clearly specified and consistent
- S3 must receive both visual data and localization data as a combined input

## Guarantees

- For each camera pose, S3 must be able to determine which visual observation it corresponds to
- S3 produces reconstruction in a **local coordinate system** — not assumed to be geographic
- Reconstruction coordinate system origin and orientation must be documented
- S3 may refine or optimize the local reconstruction; such changes do not affect the interface contract

## Failure Conditions

- If S1→S2 observation-to-pose association is broken, S3 cannot correctly associate poses with observations
- If S2's coordinate reference is unspecified or inconsistent, S3's local reconstruction will have ambiguous georeferencing
- If S2 provides poses without corresponding timestamps or frame IDs, S3 cannot establish the observation-pose correspondence

## Version

**Contract version:** 1.0.0

**Since:** Matrix S2→S3 interface establishment

## Associated Interfaces

- **S2 Output:** See [S2 → S3 Interface](system.md#6-s2-s3-interface) in system architecture
- **S3 Output:** See [S3 → S4 Interface](system.md#8-s3-s4-interface) in system architecture

## Change Log

| Version | Date | Change |
| --- | --- | --- |
| 1.0.0 | — | Initial contract definition |