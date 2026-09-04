"""
georeferencing.georeferencer

S4 — Georeferencing & Validation subsystem
Component 6: Georeferencing Pipeline Orchestration & Application Delivery

Coordinates ingestion, coordinate reference verification, 7-parameter Helmert fitting
with robust RANSAC, full reconstruction transformation, accuracy validation,
known limitations detection, and downstream S4 -> S5 contract delivery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

from .control_points import ControlPoints
from .crs import CoordinateReference
from .helmert import HelmertTransform
from .input import ReconstructionInput
from .validator import GeoreferencingValidator, ValidationResult


@dataclass
class GeoreferencedResult:
    """Standard result artifact produced by the S4 georeferencing pipeline.

    Attributes:
        points: Transformed Nx3 point cloud in the target world/geographic coordinate system.
        source_crs: Source coordinate reference descriptor.
        target_crs: Target coordinate reference descriptor.
        transformation: Fitted 3D Helmert similarity transformation.
        colors: Optional Nx3 uint8 RGB array preserved from input.
        validation_result: Optional validation result metrics from GCP accuracy assessment.
        known_limitations: List of identified constraints, caveats, or spatial limitations.
        quality_status: Quality status bundle (confidence_level, issues_detected, recommended_actions).
        metadata: Comprehensive metadata dictionary.
    """

    points: np.ndarray
    source_crs: CoordinateReference
    target_crs: CoordinateReference
    transformation: HelmertTransform
    colors: Optional[np.ndarray] = None
    validation_result: Optional[ValidationResult] = None
    known_limitations: List[str] = field(default_factory=list)
    quality_status: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert georeferenced result summary and metadata to a dictionary."""
        return {
            "num_points": int(self.points.shape[0]),
            "source_crs": self.source_crs.metadata_dict,
            "target_crs": self.target_crs.metadata_dict,
            "transformation": self.transformation.parameters(),
            "has_colors": self.colors is not None,
            "known_limitations": list(self.known_limitations),
            "quality_status": dict(self.quality_status),
            "validation": (
                self.validation_result.to_dict() if self.validation_result else None
            ),
            "metadata": dict(self.metadata),
        }

    def to_json(
        self,
        filepath: Optional[Union[str, Path]] = None,
        indent: int = 2,
    ) -> str:
        """Export result metadata and quality metrics to JSON."""
        json_str = json.dumps(self.to_dict(), indent=indent)
        if filepath is not None:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(json_str)
        return json_str

    def export_contract_payload(self) -> Dict[str, Any]:
        """Export result formatted according to the S4 -> S5 Interface Contract.

        Conforms to docs/architecture/contracts/georeferencing-application.md.
        """
        if self.points.shape[0] > 0:
            scene_origin = np.mean(self.points, axis=0).tolist()
        else:
            scene_origin = self.transformation.translation.tolist()

        val_metrics = {}
        if self.validation_result:
            val_dict = self.validation_result.to_dict()
            val_metrics = {
                "geometric_accuracy": val_dict["metrics"]["rmse_3d"],
                "horizontal_accuracy": val_dict["metrics"]["horizontal_rmse"],
                "vertical_accuracy": val_dict["metrics"]["vertical_rmse"],
                "completeness": 1.0,
                "spatial_consistency": (
                    val_dict["spatial_consistency"]["spatial_consistency_score"]
                    if val_dict.get("spatial_consistency")
                    else 1.0
                ),
                "reprojection_error": val_dict["metrics"]["mean_error"],
                "quality_score": (
                    0.95 if self.quality_status.get("confidence_level") == "high" else 0.80
                ),
            }

        return {
            "geo_referenced_scene": {
                "point_cloud": {
                    "num_points": int(self.points.shape[0]),
                    "has_colors": self.colors is not None,
                },
                "mesh": None,
                "scene_origin": scene_origin,
                "scene_orientation": self.transformation.rotation.tolist(),
                "reference_frame": self.target_crs.name or f"EPSG:{self.target_crs.epsg}",
            },
            "validation_metrics": val_metrics,
            "coordinate_reference": {
                "name": self.target_crs.name,
                "epsg": self.target_crs.epsg,
                "units": self.target_crs.units,
                "frame_type": self.target_crs.frame_type,
                "datum": self.target_crs.datum,
                "projection": self.target_crs.projection,
            },
            "quality_status": self.quality_status,
            "known_limitations": self.known_limitations,
        }


class Georeferencer:
    """High-level S4 Georeferencing and Validation Pipeline."""

    def __init__(
        self,
        reconstruction_data: ReconstructionInput,
        control_points: ControlPoints,
        source_crs: CoordinateReference,
        target_crs: CoordinateReference,
    ) -> None:
        """Initialize the georeferencer.

        Parameters:
            reconstruction_data: Reconstructed local 3D point cloud from S3.
            control_points: Ground Control Points (GCPs) correspondences.
            source_crs: Coordinate reference for the local reconstruction.
            target_crs: Coordinate reference for the target world frame.
        """
        if not isinstance(reconstruction_data, ReconstructionInput):
            raise TypeError("reconstruction_data must be a ReconstructionInput instance.")
        if not isinstance(control_points, ControlPoints):
            raise TypeError("control_points must be a ControlPoints instance.")
        if not isinstance(source_crs, CoordinateReference):
            raise TypeError("source_crs must be a CoordinateReference instance.")
        if not isinstance(target_crs, CoordinateReference):
            raise TypeError("target_crs must be a CoordinateReference instance.")

        self._reconstruction = reconstruction_data
        self._control_points = control_points
        self._source_crs = source_crs
        self._target_crs = target_crs
        self._geodetic_warning: Optional[str] = None
        self._validate_crs_contract()

        self._transform: Optional[HelmertTransform] = None

    @property
    def transformation(self) -> Optional[HelmertTransform]:
        return self._transform

    def _validate_crs_contract(self) -> None:
        """Enforce strict frame semantics and safety guardrails."""
        source = self._source_crs
        target = self._target_crs

        if source.is_local and target.is_geodetic:
            if not source.allow_local_to_world and not target.allow_local_to_world:
                raise ValueError(
                    "Cannot georeference a local source CRS into a geographic or "
                    "projected target CRS without explicit local-to-world metadata. "
                    "Set allow_local_to_world=True on the participating CRS objects."
                )

        if source.is_geodetic and target.is_local:
            if not source.allow_local_to_world and not target.allow_local_to_world:
                raise ValueError(
                    "Cannot map a geodetic source CRS into a local target frame "
                    "without explicit local-to-world metadata. "
                    "Set allow_local_to_world=True explicitly."
                )

        if source.is_geodetic and target.is_geodetic:
            if source.epsg is not None and target.epsg is not None and source.epsg != target.epsg:
                self._geodetic_warning = (
                    "Source and target CRS codes differ; 3D similarity transform alignment "
                    "is fitted over control points without full non-linear ellipsoidal datum shift."
                )

    def fit(self, max_iterations: int = 500, outlier_threshold: float = 3.0) -> HelmertTransform:
        """Fit 7-parameter Helmert transformation using robust RANSAC."""
        self._transform = HelmertTransform.from_control_points(
            self._control_points,
            max_iterations=max_iterations,
            outlier_threshold=outlier_threshold,
        )
        return self._transform

    def transform(self) -> np.ndarray:
        """Apply the fitted transformation to the full reconstruction point cloud."""
        if self._transform is None:
            raise ValueError("Helmert transformation has not been fitted. Call fit() first.")
        return self._transform.transform_points(self._reconstruction.points_array)

    def georeference(
        self,
        validate_accuracy: bool = True,
        tolerance: Optional[float] = None,
        horizontal_tolerance: Optional[float] = None,
        vertical_tolerance: Optional[float] = None,
        check_spatial_consistency: bool = True,
    ) -> GeoreferencedResult:
        """Execute the end-to-end S4 georeferencing pipeline."""
        if self._transform is None:
            self.fit()

        transformed_points = self.transform()

        colors = None
        if self._reconstruction.colors_array is not None:
            colors = np.array(self._reconstruction.colors_array, dtype=np.uint8, copy=True)

        val_result = None
        if validate_accuracy:
            validator = GeoreferencingValidator(
                control_points=self._control_points,
                transformation=self._transform,
                tolerance=tolerance,
                horizontal_tolerance=horizontal_tolerance,
                vertical_tolerance=vertical_tolerance,
            )
            src_pts = self._reconstruction.points_array if check_spatial_consistency else None
            val_result = validator.validate(
                reconstruction_points=transformed_points if check_spatial_consistency else None,
                source_points=src_pts,
            )

        limitations = self._detect_known_limitations(val_result)
        quality_status = self._assess_quality_status(val_result, limitations)

        metadata = dict(self._reconstruction.metadata_dict)
        metadata["georeferencing_method"] = "3D Similarity / Helmert (Umeyama SVD + RANSAC)"
        metadata["source_crs"] = self._crs_metadata(self._source_crs)
        metadata["target_crs"] = self._crs_metadata(self._target_crs)
        metadata["coordinate_frame_mode"] = self._coordinate_frame_mode()
        metadata["known_limitations"] = limitations
        metadata["quality_status"] = quality_status

        if self._geodetic_warning is not None:
            metadata.setdefault("warnings", []).append(self._geodetic_warning)

        return GeoreferencedResult(
            points=transformed_points,
            source_crs=self._source_crs,
            target_crs=self._target_crs,
            transformation=self._transform,
            colors=colors,
            validation_result=val_result,
            known_limitations=limitations,
            quality_status=quality_status,
            metadata=metadata,
        )

    def _detect_known_limitations(
        self,
        validation: Optional[ValidationResult],
    ) -> List[str]:
        """Automatically identify and document spatial constraints and limitations."""
        limitations: List[str] = []
        n_gcps = self._control_points.number_of_points

        if n_gcps < 5:
            limitations.append(
                f"Small number of Ground Control Points (N={n_gcps} < 5); "
                f"spatial accuracy may degrade in regions far from GCP coverage."
            )

        src = self._control_points.source_array
        xy_span = float(np.max(np.ptp(src[:, :2], axis=0)))
        z_span = float(np.ptp(src[:, 2]))
        if xy_span > 0 and (z_span / xy_span) < 0.05:
            limitations.append(
                f"Low vertical GCP distribution (vertical spread {z_span:.2f}m vs {xy_span:.2f}m horizontal); "
                f"vertical scale and tilt uncertainty is elevated."
            )

        if self._transform and self._transform.inlier_mask is not None:
            num_outliers = int(np.sum(~self._transform.inlier_mask))
            if num_outliers > 0:
                limitations.append(
                    f"{num_outliers} of {n_gcps} GCPs were rejected as outliers during RANSAC estimation."
                )

        if self._transform:
            scale_dev = abs(self._transform.scale - 1.0)
            if scale_dev > 0.08:
                limitations.append(
                    f"Significant scale transformation factor ({self._transform.scale:.4f}); "
                    f"verify metric calibration of local reconstruction."
                )

        if self._geodetic_warning:
            limitations.append(self._geodetic_warning)

        if validation and validation.rmse > 1.0:
            limitations.append(
                f"Elevated GCP residual RMSE ({validation.rmse:.3f} m); check GCP survey accuracy."
            )

        return limitations

    def _assess_quality_status(
        self,
        validation: Optional[ValidationResult],
        limitations: List[str],
    ) -> Dict[str, Any]:
        """Determine overall confidence level and recommended actions."""
        issues = list(limitations)
        recommended_actions: List[str] = []

        if validation and validation.rmse <= 0.3 and len(limitations) == 0:
            confidence = "high"
        elif validation and (validation.rmse > 1.0 or len(limitations) >= 3):
            confidence = "low"
            recommended_actions.append("Survey additional GCPs across full spatial and vertical extent of area.")
        else:
            confidence = "medium"
            if len(limitations) > 0:
                recommended_actions.append("Review GCP residuals and verify coordinate reference system datum.")

        return {
            "confidence_level": confidence,
            "issues_detected": issues,
            "recommended_actions": recommended_actions,
        }

    def _coordinate_frame_mode(self) -> str:
        if self._source_crs.is_local and self._target_crs.is_geodetic:
            return "explicit_local_to_world"
        if self._source_crs.is_geodetic and self._target_crs.is_local:
            return "explicit_world_to_local"
        if self._source_crs.is_geodetic and self._target_crs.is_geodetic:
            return "world_similarity"
        return "local_similarity"

    @staticmethod
    def _crs_metadata(crs: CoordinateReference) -> Dict[str, Any]:
        return {
            "name": crs.name,
            "epsg": crs.epsg,
            "units": crs.units,
            "dimension": crs.dimension,
            "description": crs.description,
            "frame_type": crs.frame_type,
            "datum": crs.datum,
            "projection": crs.projection,
            "allow_local_to_world": crs.allow_local_to_world,
            "metadata": crs.metadata_dict,
        }
