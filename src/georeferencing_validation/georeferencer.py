"""
S4 — Georeferencing & Validation

High-level orchestration of the georeferencing pipeline.

This module coordinates:
    - ReconstructionInput
    - ControlPoints
    - CoordinateReference
    - HelmertTransform

It does not implement Helmert mathematics or validation metrics.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from .input import ReconstructionInput
from .control_points import ControlPoints
from .crs import CoordinateReference
from .helmert import HelmertTransform


@dataclass
class GeoreferencedResult:
    """
    Result produced by the S4 georeferencing pipeline.

    Attributes
    ----------
    points : np.ndarray
        Transformed Nx3 points in the target/reference coordinate frame.
    source_crs : CoordinateReference
        Source/local coordinate-reference descriptor.
    target_crs : CoordinateReference
        Target/reference coordinate-reference descriptor.
    transformation : HelmertTransform
        Fitted 3D similarity transformation mapping source -> target.
    colors : Optional[np.ndarray]
        Optional Nx3 uint8 color array preserved from the input reconstruction.
    metadata : Dict[str, Any]
        Metadata associated with the georeferenced reconstruction.
    """

    points: np.ndarray
    source_crs: CoordinateReference
    target_crs: CoordinateReference
    transformation: HelmertTransform
    colors: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Georeferencer:
    """
    High-level S4 georeferencing pipeline.

    The pipeline is:

        ReconstructionInput
                |
                v
        ControlPoints
                |
                v
        HelmertTransform
                |
                v
        Georeferenced point cloud

    The Helmert transformation is estimated from the supplied control
    points and then applied to the complete reconstructed point cloud.
    """

    def __init__(
        self,
        reconstruction_data: ReconstructionInput,
        control_points: ControlPoints,
        source_crs: CoordinateReference,
        target_crs: CoordinateReference,
    ) -> None:
        """
        Initialize the georeferencer.

        Parameters
        ----------
        reconstruction_data : ReconstructionInput
            Reconstructed local 3D point cloud received from S3.
        control_points : ControlPoints
            Corresponding source and target control points.
        source_crs : CoordinateReference
            Coordinate-reference descriptor for the reconstruction.
        target_crs : CoordinateReference
            Coordinate-reference descriptor for the target frame.

        Raises
        ------
        TypeError
            If any argument is not the expected S4 class.
        """
        if not isinstance(reconstruction_data, ReconstructionInput):
            raise TypeError(
                "reconstruction_data must be a ReconstructionInput instance."
            )

        if not isinstance(control_points, ControlPoints):
            raise TypeError(
                "control_points must be a ControlPoints instance."
            )

        if not isinstance(source_crs, CoordinateReference):
            raise TypeError(
                "source_crs must be a CoordinateReference instance."
            )

        if not isinstance(target_crs, CoordinateReference):
            raise TypeError(
                "target_crs must be a CoordinateReference instance."
            )

        self._reconstruction = reconstruction_data
        self._control_points = control_points
        self._source_crs = source_crs
        self._target_crs = target_crs
        self._validate_crs_contract()

        # None until fit() is called.
        self._transform: Optional[HelmertTransform] = None

    @property
    def transformation(self) -> Optional[HelmertTransform]:
        """Return the fitted Helmert transformation."""
        return self._transform

    def _validate_crs_contract(self) -> None:
        """Guard against silently labeling a local reconstruction as a world CRS.

        A local point cloud transformed into a geographic or projected world
        frame is only valid when this is explicitly declared through the CRS
        metadata. This prevents false claims of real geodetic georeferencing.
        """

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
                    "Source and target CRS codes differ; this implementation only "
                    "fits a local 3D similarity transform and does not perform a "
                    "full CRS conversion."
                )
            else:
                self._geodetic_warning = None
        else:
            self._geodetic_warning = None

    def fit(self) -> HelmertTransform:
        """Estimate the 7-parameter Helmert transformation."""
        self._transform = HelmertTransform.from_control_points(
            self._control_points
        )
        return self._transform

    def transform(self) -> np.ndarray:
        """Apply the fitted Helmert transformation to the reconstruction."""
        if self._transform is None:
            raise ValueError(
                "Helmert transformation has not been fitted. "
                "Call fit() first."
            )

        points = self._reconstruction.points_array
        return self._transform.transform_points(points)

    def georeference(self) -> GeoreferencedResult:
        """Execute the complete S4 georeferencing pipeline."""
        if self._transform is None:
            self.fit()

        transformed_points = self.transform()

        # Preserve colors without modifying the original reconstruction.
        colors: Optional[np.ndarray] = None

        if self._reconstruction.colors_array is not None:
            colors = np.array(
                self._reconstruction.colors_array,
                dtype=np.uint8,
                copy=True,
            )

        # Copy original metadata so the input object is never mutated.
        metadata: Dict[str, Any] = dict(
            self._reconstruction.metadata_dict
        )

        metadata["georeferencing_method"] = "3D Similarity / Helmert"
        metadata["source_crs"] = self._crs_metadata(
            self._source_crs
        )
        metadata["target_crs"] = self._crs_metadata(
            self._target_crs
        )
        metadata["coordinate_frame_mode"] = self._coordinate_frame_mode()
        if self._geodetic_warning is not None:
            metadata["warnings"] = metadata.get("warnings", [])
            metadata["warnings"].append(self._geodetic_warning)

        return GeoreferencedResult(
            points=transformed_points,
            source_crs=self._source_crs,
            target_crs=self._target_crs,
            transformation=self._transform,
            colors=colors,
            metadata=metadata,
        )

    def _coordinate_frame_mode(self) -> str:
        """Describe the transformation mode being used.

        The implementation is intentionally conservative: a local-to-world map is
        only allowed when the CRS metadata explicitly declares the permission.
        """

        if self._source_crs.is_local and self._target_crs.is_geodetic:
            return "explicit_local_to_world"
        if self._source_crs.is_geodetic and self._target_crs.is_local:
            return "explicit_world_to_local"
        if self._source_crs.is_geodetic and self._target_crs.is_geodetic:
            return "world_similarity"
        return "local_similarity"

    @staticmethod
    def _crs_metadata(
        crs: CoordinateReference,
    ) -> Dict[str, Any]:
        """Convert CRS information into a simple metadata dictionary."""
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
