"""S3 — 3D Reconstruction: contracts.

Defines the Pydantic ``S3Contract`` that constitutes the **S3 → S4
boundary** as documented in
``docs/architecture/contracts/reconstruction-georeferencing.md``.

The contract carries the reconstructed point cloud as a base64-encoded
or filepath-referenced PLY (we use a path here, matching the long-
standing orchestrator behavior), the spatial reference describing the
local coordinate frame, and the reconstruction quality metadata.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .models.s3_output import (
    PointCloudData,
    ReconstructionQuality,
    S3ReconstructionResult,
    SpatialReference,
)


class S3Contract(BaseModel):
    """Canonical S3 → S4 boundary payload."""

    schema_version: str = "1.0.0"
    scene_id: str = "scene_001"
    job_id: Optional[str] = None
    status: str = "success"

    point_cloud: Optional[Dict[str, Any]] = None
    spatial_reference: Optional[Dict[str, Any]] = None
    quality: Optional[Dict[str, Any]] = None

    artifact_paths: Dict[str, str] = Field(default_factory=dict)
    failure_info: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


def s3_result_to_contract(
    result: S3ReconstructionResult,
    artifact_paths: Optional[Dict[str, str]] = None,
) -> S3Contract:
    """Convert an internal :class:`S3ReconstructionResult` to :class:`S3Contract`.

    The orchestrator is the only caller; this keeps S3's in-memory types
    encapsulated inside ``_internal``.
    """
    pcd = result.point_cloud
    pc_payload: Optional[Dict[str, Any]] = None
    if pcd is not None:
        pc_payload = {
            "num_points": int(pcd.num_points),
            "has_colors": pcd.colors is not None,
            "has_normals": pcd.normals is not None,
        }

    sr_payload: Optional[Dict[str, Any]] = None
    if result.spatial_reference is not None:
        sr_payload = {
            "coordinate_frame": result.spatial_reference.coordinate_frame,
            "units": result.spatial_reference.units,
            "reference_system": result.spatial_reference.reference_system,
        }

    q_payload: Optional[Dict[str, Any]] = None
    if result.quality is not None:
        q_payload = {
            "input_observations_count": result.quality.input_observations_count,
            "processed_observations_count": result.quality.processed_observations_count,
            "triangulated_tracks_count": result.quality.triangulated_tracks_count,
            "triangulation_success_ratio": result.quality.triangulation_success_ratio,
            "mean_reprojection_error_px": result.quality.mean_reprojection_error_px,
            "median_reprojection_error_px": result.quality.median_reprojection_error_px,
            "coverage_ratio": result.quality.coverage_ratio,
            "processing_time_seconds": result.quality.processing_time_seconds,
        }

    return S3Contract(
        schema_version="1.0.0",
        scene_id=result.scene_id,
        job_id=result.job_id,
        status=str(result.status),
        point_cloud=pc_payload,
        spatial_reference=sr_payload,
        quality=q_payload,
        artifact_paths=dict(artifact_paths or {}),
        failure_info=result.failure_info,
        metadata=dict(result.metadata or {}),
    )


__all__ = ["S3Contract", "s3_result_to_contract"]
