# S2 → S3 Interface Contract

## Overview

S2 (Localization & Sensor Fusion) provides both the original visual observations from S1 and the camera localization data (poses, trajectory) to S3 (3D Reconstruction) for multi-view 3D reconstruction.

S3 also accepts an **optional external camera calibration** as a first-class input — distinct from the pose data that S2 produces. See [§ Camera Calibration](#camera-calibration) below.

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

### Camera (per-observation, forwarded from S1 → S2)

| Field | Type | Description |
| --- | --- | --- |
| `width` | `int` | Image width in pixels |
| `height` | `int` | Image height in pixels |
| `intrinsics` | `object?` | `fx, fy, cx, cy` (may be null when no calibration is available) |
| `distortion` | `object?` | `{model, coefficients}` (may be null) |

The per-observation camera block is **not** an external calibration source — it is forwarded from S1 and is what S3 falls back on when no external calibration is provided. The external calibration input below supersedes it.

## Camera Calibration

External camera calibration is an **optional** S3 input, supplied alongside the S2 contract.

| Property | Value |
| :--- | :--- |
| Loader | `src.reconstruction.OpenCVCameraCalibrationLoader` |
| Format | OpenCV ``%YAML:1.0`` YAML with ``!!opencv-matrix`` / ``!x!opencv-matrix`` tags |
| Required fields | `camera_name`, `image_width`, `image_height`, `distortion_model`, `camera_matrix`, `distortion_coefficients` |
| Supported distortion models | `plumb_bob` (alias `radtan`), 5 coefficients `[k1, k2, p1, p2, k3]` |
| Resolution matching | calibration `image_width × image_height` must match the video; if not, the caller must explicitly call `CameraCalibration.scale_to_resolution(target_w, target_h)` which enforces isotropic uniform scaling |
| Validation | structural — 3×3 `camera_matrix` with `K[2,2] ≈ 1.0`, `fx/fy > 0`, principal point inside image, non-finite value rejection |
| Effect on triangulation | observation pixel coordinates are undistorted via `cv2.undistortPoints(P=K)` before being fed to the triangulator |

**Calibration vs. Pose — separation of concerns**

Calibration answers: *how does a pixel map to a camera ray?*

Pose answers: *where is that ray pointing in the world?*

Both are required for correct triangulation. The calibration input does **not** alter camera poses; the pose input does **not** alter calibration.

**Raw vs undistorted observations**

* `S2Observation.features[*].xy` always carries **raw** (distorted) pixel coordinates — that is what the upstream detector actually saw.
* S3's preparer produces a `PreparedTrack` with both `points_2d_raw` and `points_2d` (undistorted when a calibration is supplied).
* The triangulation engine consumes `points_2d` (the undistorted coordinates), keeping the `P = K [R | t]` projection model mathematically consistent.

**Failure modes**

* Malformed YAML or missing required fields → `CameraCalibrationError` with a descriptive message; the pipeline halts before S3.
* Resolution mismatch → `CameraCalibrationError` raised by `S3ReconstructionPipeline.run`; no silent reuse.
* All-zero distortion coefficients → undistortion is a no-op; `undistortion_applied` reports `false`.

## Outputs

S3 generates a 3D reconstruction including point cloud, mesh, and associated metadata.

When a calibration was supplied, the output `metadata.json` includes:

```json
{
  "metadata": {
    "camera_calibration": {
      "camera_name": "DJI_Air_2S",
      "image_width": 1920,
      "image_height": 1080,
      "distortion_model": "plumb_bob",
      "camera_matrix": [...],
      "distortion_coefficients": [...],
      "source": "path/to/camera_calibration.yaml",
      "undistortion_applied": true
    },
    "camera_calibration_summary": {
      "camera_name": "DJI_Air_2S",
      "image_width": 1920,
      "image_height": 1080,
      "distortion_model": "plumb_bob",
      "distortion_applied": true
    }
  }
}
```

## Preconditions

- Each S2 localization result must be associable to an S1 observation via `frame_id`, `timestamp`, or index
- The coordinate reference system used by S2 must be clearly specified and consistent
- S3 must receive both visual data and localization data as a combined input
- If an external calibration is supplied, its resolution must match the actual video resolution (after explicit scaling via `scale_to_resolution`)

## Guarantees

- For each camera pose, S3 must be able to determine which visual observation it corresponds to
- S3 produces reconstruction in a **local coordinate system** — not assumed to be geographic
- Reconstruction coordinate system origin and orientation must be documented
- S3 may refine or optimize the local reconstruction; such changes do not affect the interface contract
- When a calibration is supplied, observation pixel coordinates are undistorted with `cv2.undistortPoints(P=K)` so the existing `P = K [R | t]` projection model remains consistent. The raw (distorted) coordinates are preserved on each `PreparedTrack` for audit and diagnostics.

## Failure Conditions

- If S1→S2 observation-to-pose association is broken, S3 cannot correctly associate poses with observations
- If S2's coordinate reference is unspecified or inconsistent, S3's local reconstruction will have ambiguous georeferencing
- If S2 provides poses without corresponding timestamps or frame IDs, S3 cannot establish the observation-pose correspondence
- If an external calibration file is malformed, resolution-mismatched, or uses an unsupported distortion model, S3 raises `CameraCalibrationError` and does not silently fall back

## Version

**Contract version:** 1.1.0

**Since:** Matrix S2→S3 interface establishment

## Associated Interfaces

- **S2 Output:** See [S2 → S3 Interface](system.md#6-s2-s3-interface) in system architecture
- **S3 Output:** See [S3 → S4 Interface](system.md#8-s3-s4-interface) in system architecture

## Change Log

| Version | Date | Change |
| --- | --- | --- |
| 1.0.0 | — | Initial contract definition |
| 1.1.0 | — | Added optional **Camera Calibration** input; documented undistortion, resolution-compatibility policy, raw vs undistorted observation separation, and `metadata.camera_calibration` output fields. |
