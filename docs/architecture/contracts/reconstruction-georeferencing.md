# S3 → S4 Interface Contract

## Overview

S3 (3D Reconstruction) provides the 3D reconstruction and associated metadata to S4 (Georeferencing & Validation) for geographic alignment and spatial validation.

## Inputs

### 3D Reconstruction from S3

| Field | Type | Description |
| --- | --- | --- |
| `point_cloud` | `PointCloud` | Generated point cloud data |
| `mesh` | `Mesh` | Generated mesh (optional, may be absent) |
| `reconstruction_metadata` | `object` | Overall reconstruction metadata |
| `spatial_reference` | `object` | Local coordinate reference system description |
| `quality_information` | `object` | Reconstruction quality metrics and assessments |

### PointCloud

| Field | Type | Description |
| --- | --- | --- |
| `points` | `array<Point>` | List of 3D points |
| `colors` | `array<Color>` | Optional per-point RGB colors |
| `normals` | `array<Normal>` | Optional per-point normals |

### Point

| Field | Type | Description |
| --- | --- | --- |
| `x`, `y`, `z` | `float` | Coordinates in local reconstruction system |
| `point_id` | `string` | Unique identifier for the point |

### Mesh

| Field | Type | Description |
| --- | --- | --- |
| `vertices` | `array<Vertex>` | 3D vertices |
| `faces` | `array<Face>` | Triangle face indices |
| `textures` | `object` | Optional texture mapping data |

### Vertex

| Field | Type | Description |
| --- | --- | --- |
| `x`, `y`, `z` | `float` | Vertex coordinates in local system |

### Spatial Reference (from S3)

| Field | Type | Description |
| --- | --- | --- |
| `coordinate_frame` | `string` | Description of the local frame (e.g., "COLMAP local frame") |
| `units` | `string` | Unit of measurement (e.g., "meters") |
| `origin` | `array<float>[3]` | Origin position in local coordinates |
| `orientation` | `array<float>[3]` or `quaternion` | Frame orientation |
| `reference_system` | `string` | Explicit statement that this is local, not geographic |

### Quality Information

| Field | Type | Description |
| --- | --- | --- |
| `point_count` | `int` | Total number of points in reconstruction |
| `mesh_count` | `int` | Number of meshes generated |
| `reprojection_error` | `float` | Mean reprojection error (if available) |
| `triangulation_ratio` | `float` | Ratio of triangulated points |

## Outputs

S4 returns a georeferenced and validated 3D scene with geographic alignment and quality metrics.

## Preconditions

- S3 must produce a valid reconstruction in its local coordinate system
- S3 must document the local coordinate frame, origin, and orientation
- S3 must not claim geographic coordinates unless explicitly supported

## Guarantees

- S4 transforms the local reconstruction into geographic/world coordinates
- S4 provides validation metrics assessing reconstruction spatial quality
- S4 documents known limitations of the georeferencing
- Output coordinate system, units, and reference frame are explicitly specified

## Failure Conditions

- If S3's local coordinate system is unspecified, S4 cannot perform geographic alignment
- If S3 claims geographic coordinates that are actually local, S4 georeferencing will produce incorrect results
- If reconstruction quality is very poor, georeferencing may fail or produce unreliable results

## Version

**Contract version:** 1.0.0

**Since:** Matrix S3→S4 interface establishment

## Associated Interfaces

- **S3 Output:** See [S3 → S4 Interface](system.md#8-s3-s4-interface) in system architecture
- **S4 Output:** See [S4 → S5 Interface](system.md#10-s4-s5-interface) in system architecture

## Change Log

| Version | Date | Change |
| --- | --- | --- |
| 1.0.0 | — | Initial contract definition |