"""
S3 Reconstruction Output Models

Minimal dataclasses consumed by the S3 output packager, quality evaluator,
and the S4 boundary conversion (to_s4_reconstruction_input).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class BoundingBox3D:
    """Axis-aligned 3D bounding box."""

    min_point: np.ndarray
    max_point: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_point": np.asarray(self.min_point).tolist(),
            "max_point": np.asarray(self.max_point).tolist(),
        }


@dataclass
class PointCloudData:
    """N x 3 (and optional N x 3 attributes) point cloud."""

    points: np.ndarray
    colors: Optional[np.ndarray] = None
    normals: Optional[np.ndarray] = None
    confidences: Optional[np.ndarray] = None

    @property
    def num_points(self) -> int:
        if self.points is None:
            return 0
        return int(np.asarray(self.points).shape[0])

    @property
    def bounding_box(self) -> BoundingBox3D:
        if self.num_points == 0:
            zeros = np.zeros(3, dtype=np.float64)
            return BoundingBox3D(min_point=zeros, max_point=zeros)
        pts = np.asarray(self.points, dtype=np.float64)
        return BoundingBox3D(min_point=pts.min(axis=0), max_point=pts.max(axis=0))


@dataclass
class ReconstructionQuality:
    """Quality summary for a reconstruction result."""

    input_observations_count: int = 0
    processed_observations_count: int = 0
    triangulated_tracks_count: int = 0
    triangulation_success_ratio: float = 0.0
    mean_reprojection_error_px: float = 0.0
    median_reprojection_error_px: float = 0.0
    coverage_ratio: float = 0.0
    processing_time_seconds: float = 0.0


@dataclass
class SpatialReference:
    """Describes the coordinate reference system of a reconstruction."""

    coordinate_frame: str = "S3_LOCAL"
    units: str = "meters"
    origin: Optional[Any] = None
    orientation: Optional[Any] = None
    reference_system: str = "local"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class S3ReconstructionResult:
    """Top-level S3 reconstruction result consumed by S4."""

    scene_id: str = "scene_001"
    job_id: Optional[str] = None
    status: Any = "success"
    point_cloud: Optional[PointCloudData] = None
    spatial_reference: Optional[SpatialReference] = None
    quality: Optional[ReconstructionQuality] = None
    failure_info: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_metadata_dict(self) -> Dict[str, Any]:
        """Serialize the result into a JSON-friendly metadata dict."""
        bbox = self.point_cloud.bounding_box if self.point_cloud else None
        return {
            "scene_id": self.scene_id,
            "job_id": self.job_id,
            "status": str(self.status),
            "num_points": self.point_cloud.num_points if self.point_cloud else 0,
            "bounding_box": bbox.to_dict() if bbox is not None else None,
            "spatial_reference": (
                {
                    "coordinate_frame": self.spatial_reference.coordinate_frame,
                    "units": self.spatial_reference.units,
                }
                if self.spatial_reference
                else None
            ),
            "quality": (
                {
                    "input_observations_count": self.quality.input_observations_count,
                    "processed_observations_count": self.quality.processed_observations_count,
                    "triangulated_tracks_count": self.quality.triangulated_tracks_count,
                    "triangulation_success_ratio": self.quality.triangulation_success_ratio,
                    "mean_reprojection_error_px": self.quality.mean_reprojection_error_px,
                    "median_reprojection_error_px": self.quality.median_reprojection_error_px,
                    "coverage_ratio": self.quality.coverage_ratio,
                    "processing_time_seconds": self.quality.processing_time_seconds,
                }
                if self.quality
                else None
            ),
            "failure_info": self.failure_info,
            "metadata": self.metadata,
        }

    def to_s4_reconstruction_input(self):
        """Deprecated: S4 boundary conversion is now the orchestrator's job.

        Kept as a stub that raises ``NotImplementedError`` so that any
        legacy caller fails loudly instead of silently re-introducing a
        cross-subsystem import. S3 → S4 conversion goes through the
        S3 → S4 wire-format contract handled by ``run_s4`` (which
        reads S3's ``scene.ply`` + ``metadata.json`` directly).
        """
        raise NotImplementedError(
            "S3ReconstructionResult.to_s4_reconstruction_input has been "
            "removed during subsystem isolation. Use S4's run_s4() with "
            "the S3Contract instead."
        )
