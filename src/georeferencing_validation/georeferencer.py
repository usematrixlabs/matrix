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

        # None until fit() is called.
        self._transform: Optional[HelmertTransform] = None

    @property
    def transformation(self) -> Optional[HelmertTransform]:
        """
        Return the fitted Helmert transformation.

        Returns
        -------
        Optional[HelmertTransform]
            The fitted transformation, or None if fit() has not
            been called yet.
        """
        return self._transform

    def fit(self) -> HelmertTransform:
        """
        Estimate the 7-parameter Helmert transformation.

        The actual transformation estimation is delegated to
        HelmertTransform.from_control_points().

        Returns
        -------
        HelmertTransform
            The fitted source-to-target transformation.

        Raises
        ------
        ValueError
            If the control points are unsuitable for estimating
            the transformation.
        """
        self._transform = HelmertTransform.from_control_points(
            self._control_points
        )

        return self._transform

    def transform(self) -> np.ndarray:
        """
        Apply the fitted Helmert transformation to the reconstruction.

        Returns
        -------
        np.ndarray
            Transformed Nx3 point cloud with dtype float64.

        Raises
        ------
        ValueError
            If the transformation has not been fitted.
        """
        if self._transform is None:
            raise ValueError(
                "Helmert transformation has not been fitted. "
                "Call fit() first."
            )

        points = self._reconstruction.points_array

        return self._transform.transform_points(points)

    def georeference(self) -> GeoreferencedResult:
        """
        Execute the complete S4 georeferencing pipeline.

        If the transformation has not already been fitted, it is
        automatically estimated from the control points.

        Returns
        -------
        GeoreferencedResult
            Complete georeferenced reconstruction containing transformed
            points, CRS information, transformation parameters, optional
            colors, and metadata.
        """
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

        # Add S4 georeferencing information.
        metadata["georeferencing_method"] = (
            "3D Similarity / Helmert"
        )

        metadata["source_crs"] = self._crs_metadata(
            self._source_crs
        )

        metadata["target_crs"] = self._crs_metadata(
            self._target_crs
        )

        return GeoreferencedResult(
            points=transformed_points,
            source_crs=self._source_crs,
            target_crs=self._target_crs,
            transformation=self._transform,
            colors=colors,
            metadata=metadata,
        )

    @staticmethod
    def _crs_metadata(
        crs: CoordinateReference,
    ) -> Dict[str, Any]:
        """
        Convert CRS information into a simple metadata dictionary.

        This method intentionally does not perform CRS conversion or
        external EPSG validation.

        Parameters
        ----------
        crs : CoordinateReference
            CRS descriptor.

        Returns
        -------
        Dict[str, Any]
            Serializable CRS metadata.
        """
        return {
            "name": crs.name,
            "epsg": crs.epsg,
            "units": crs.units,
            "dimension": crs.dimension,
            "description": crs.description,
            "metadata": crs.metadata_dict,
        }
