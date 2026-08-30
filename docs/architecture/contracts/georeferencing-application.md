# S4 → S5 Interface Contract

## Overview

S4 (Georeferencing & Validation) provides the georeferenced and validated 3D scene to S5 (Application & Deployment) for user-facing interaction, visualization, and deployment.

## Inputs

### Georeferenced + Validated Scene from S4

| Field | Type | Description |
| --- | --- | --- |
| `geo_referenced_scene` | `GeoreferencedScene` | The complete georeferenced 3D scene |
| `validation_metrics` | `object` | Spatial accuracy and quality metrics |
| `coordinate_reference` | `object` | Geographic coordinate reference system |
| `quality_status` | `object` | Confidence and limitation reporting |
| `known_limitations` | `array<string>` | Known issues or constraints |

### GeoreferencedScene

| Field | Type | Description |
| --- | --- | --- |
| `point_cloud` | `PointCloud` | Georeferenced point cloud (world coordinates) |
| `mesh` | `Mesh` | Georeferenced mesh (world coordinates) |
| `scene_origin` | `array<float>[3]` | Origin position in geographic coordinates |
| `scene_orientation` | `array<float>[3]` or `quaternion` | Orientation in geographic coordinates |
| `reference_frame` | `string` | Geographic reference frame (e.g., "WGS84 / UTM zone ...") |

### ValidationMetrics

| Field | Type | Description |
| --- | --- | --- |
| `geometric_accuracy` | `float` | Estimated horizontal/vertical accuracy |
| `completeness` | `float` | Percentage of expected scene reconstructed |
| `spatial_consistency` | `float` | Internal consistency metric |
| `reprojection_error` | `float` | Post-georeferencing reprojection error |
| `quality_score` | `float` | Overall quality score (0-1) |

### QualityStatus

| Field | Type | Description |
| --- | --- | --- |
| `confidence_level` | `string` | "high", "medium", "low" |
| `issues_detected` | `array<string>` | List of identified issues |
| `recommended_actions` | `array<string>` | Suggested next steps |

## Outputs

S5 exposes the georeferenced scene to the user through the application interface, visualization, and API.

## Preconditions

- S4 must provide explicit coordinate reference system information
- S4 must not assume S5 understands internal georeferencing algorithms
- S5 must be able to present the result without needing to understand S4's internal algorithms

## Guarantees

- S5 provides user-accessible output format (API, CLI, user interface)
- S5 presents geographic coordinates in a user-friendly format
- S5 documents any limitations inherited from S4
- Output is suitable for visualization and further analysis

## Failure Conditions

- If S4's coordinate reference is incomplete or inconsistent, S5 cannot correctly display or use the georeferenced data
- If S4 does not document known limitations, S5 may present results without appropriate warnings
- If the georeferenced scene data is malformed or missing critical fields, S5 should produce a clear error rather than silently failing

## Version

**Contract version:** 1.0.0

**Since:** Matrix S4→S5 interface establishment

## Associated Interfaces

- **S4 Output:** See [S4 → S5 Interface](system.md#10-s4-s5-interface) in system architecture

## Change Log

| Version | Date | Change |
| --- | --- | --- |
| 1.0.0 | — | Initial contract definition |