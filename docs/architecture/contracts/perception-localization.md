# S1 → S2 Interface Contract

## Overview

S1 (Visual Perception) provides visual observations and available UAV sensor information to S2 (Localization & Sensor Fusion) for camera localization and trajectory estimation.

## Inputs

### Visual Observations

| Field | Type | Description |
| --- | --- | --- |
| `frames` | `array<Frame>` | Ordered list of image frames / keyframes extracted from UAV video |
| `keyframes` | `array<Keyframe>` | Subset of frames selected for feature extraction and matching |
| `frame_ordering` | `sequence` | Temporal ordering of frames as they were captured |
| `visual_metadata` | `object` | Per-frame metadata including quality scores, feature counts, etc. |

### Frame (individual)

| Field | Type | Description |
| --- | --- | --- |
| `image_path` | `string` | Path or identifier for the frame image |
| `timestamp` | `float` | Capture timestamp (monotonic, seconds) |
| `frame_id` | `string` | Unique identifier for the frame |
| `image_width` | `int` | Width in pixels |
| `image_height` | `int` | Height in pixels |
| `exposure_time` | `float` | Camera exposure time (if available) |
| `camera_id` | `string` | Camera identifier (if multiple cameras) |

### Available UAV Information (optional)

| Field | Type | Description |
| --- | --- | --- |
| `gps_coordinates` | `object` | Latitude, longitude, altitude if available |
| `gnss_status` | `string` | GNSS fix status (e.g., "fixed", "float", "none") |
| `imu_data` | `object` | Accelerometer and gyroscope measurements |
| `altitude` | `float` | Barometric or GPS altitude |
| `rtk_ppk` | `object` | RTK/PPK correction data |
| `flight_telemetry` | `object` | Flight controller state, speed, heading, etc. |
| `sensor_measurements` | `object` | Other available sensor data |

## Outputs

S2 returns camera localization results associating poses with the visual observations provided by S1.

## Preconditions

- S1 must preserve all available UAV-supplied information without silently discarding it
- Timestamp association between frames and sensor data must be maintainable
- Frame identifiers must be unique and consistent throughout the pipeline

## Guarantees

- Each localization result must include a documented association to the relevant observation via `frame_id` and/or `timestamp`
- If GPS/GNSS/IMU data is provided by S1, S2 must fuse it with visual observations
- If no additional sensor data is available, S2 must still produce localization using visual-only methods
- S2 must not assume any particular optional sensor is always present

## Failure Conditions

- If S1 discards sensor information that S2 needs, localization quality may degrade unexpectedly
- If timestamps are not associated with frames, temporal alignment with sensor data is lost
- If frame identifiers are not unique, observation-to-pose association becomes ambiguous

## Version

**Contract version:** 1.0.0

**Since:** Matrix S1→S2 interface establishment

## Associated Interfaces

- **S1 Output:** See [S1 → S2 Interface](system.md#4-s1-s2-interface) in system architecture
- **S2 Output:** See [S2 → S3 Interface](system.md#6-s2-s3-interface) in system architecture

## Change Log

| Version | Date | Change |
| --- | --- | --- |
| 1.0.0 | — | Initial contract definition |