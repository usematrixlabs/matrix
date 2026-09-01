
"""
S4 — Georeferencing & Validation

Accuracy validation for the georeferencing transformation.

This module evaluates how accurately a fitted 3D Helmert transformation
maps source control points to their known target/reference coordinates.

Responsibilities:
    - Apply the fitted Helmert transformation to source control points.
    - Calculate residuals.
    - Calculate per-point Euclidean errors.
    - Calculate RMSE and other accuracy statistics.
    - Optionally determine pass/fail against a user-supplied tolerance.

This module does NOT:
    - Estimate the Helmert transformation.
    - Perform CRS conversion.
    - Transform the complete reconstruction.
    - Modify input data.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from .control_points import ControlPoints
from .helmert import HelmertTransform


@dataclass
class ValidationResult:
    """
    Results of S4 georeferencing accuracy validation.

    Attributes
    ----------
    residuals : np.ndarray
        (N, 3) array containing transformed_source - target.
        Each row is [dx, dy, dz].

    point_errors : np.ndarray
        (N,) array containing the Euclidean error for each control point.

    rmse : float
        Root-mean-square 3D Euclidean error.

    mean_error : float
        Mean Euclidean error.

    max_error : float
        Maximum Euclidean error.

    min_error : float
        Minimum Euclidean error.

    num_points : int
        Number of control-point correspondences used.

    median_error : float
        Median Euclidean error.

    std_error : float
        Standard deviation of Euclidean errors.

    rmse_x : float
        RMSE of X-axis residuals.

    rmse_y : float
        RMSE of Y-axis residuals.

    rmse_z : float
        RMSE of Z-axis residuals.

    tolerance : Optional[float]
        User-supplied RMSE tolerance used for pass/fail evaluation.

    passed : Optional[bool]
        True if RMSE <= tolerance, False if RMSE > tolerance,
        or None if no tolerance was supplied.

    extras : Dict[str, Any]
        Additional validator information. Kept minimal for now.
    """

    residuals: np.ndarray
    point_errors: np.ndarray
    rmse: float
    mean_error: float
    max_error: float
    min_error: float
    num_points: int
    median_error: float
    std_error: float
    rmse_x: float
    rmse_y: float
    rmse_z: float
    tolerance: Optional[float]
    passed: Optional[bool]
    extras: Dict[str, Any] = field(default_factory=dict)


class GeoreferencingValidator:
    """
    Validate the accuracy of a fitted 3D Helmert transformation.

    The validation process is:

        source control points
                |
                v
        Helmert transformation
                |
                v
        predicted target points
                |
                v
        compare with known target points
                |
                v
        residuals and accuracy metrics

    Parameters
    ----------
    control_points : ControlPoints
        Validated source and target control-point correspondences.

    transformation : HelmertTransform
        Fitted Helmert transformation mapping source coordinates
        to target coordinates.

    tolerance : Optional[float]
        Optional RMSE tolerance used to determine pass/fail status.
        No default tolerance is imposed.
    """

    def __init__(
        self,
        control_points: ControlPoints,
        transformation: HelmertTransform,
        tolerance: Optional[float] = None,
    ) -> None:
        """
        Initialize the georeferencing validator.

        Parameters
        ----------
        control_points : ControlPoints
            Validated control-point correspondences.

        transformation : HelmertTransform
            Fitted source-to-target Helmert transformation.

        tolerance : Optional[float]
            Optional non-negative finite RMSE tolerance.

        Raises
        ------
        TypeError
            If control_points or transformation has an invalid type,
            or tolerance cannot be converted to a number.

        ValueError
            If tolerance is negative or non-finite.
        """
        if not isinstance(control_points, ControlPoints):
            raise TypeError(
                "control_points must be a ControlPoints instance."
            )

        if not isinstance(transformation, HelmertTransform):
            raise TypeError(
                "transformation must be a HelmertTransform instance."
            )

        if tolerance is not None:
            try:
                tolerance_value = float(tolerance)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "tolerance must be numeric when provided."
                ) from exc

            if not np.isfinite(tolerance_value):
                raise ValueError(
                    "tolerance must be finite."
                )

            if tolerance_value < 0.0:
                raise ValueError(
                    "tolerance must be non-negative."
                )

            self._tolerance: Optional[float] = tolerance_value
        else:
            self._tolerance = None

        self._control_points = control_points
        self._transformation = transformation

    @property
    def tolerance(self) -> Optional[float]:
        """Return the configured RMSE tolerance."""
        return self._tolerance

    def validate(self) -> ValidationResult:
        """
        Validate the fitted Helmert transformation.

        The source control points are transformed using the supplied
        Helmert transformation. The transformed points are then compared
        against the known target control points.

        Returns
        -------
        ValidationResult
            Residuals and aggregated accuracy metrics.

        Raises
        ------
        ValueError
            If the control-point data or calculated results contain
            invalid dimensions or non-finite values.
        """
        source = np.asarray(
            self._control_points.source_array,
            dtype=np.float64,
        )

        target = np.asarray(
            self._control_points.target_array,
            dtype=np.float64,
        )

        self._validate_control_point_arrays(source, target)

        # Apply the already-fitted Helmert transformation.
        transformed = self._transformation.transform_points(source)

        if transformed.shape != source.shape:
            raise ValueError(
                "transformed control points have an unexpected shape."
            )

        if not np.all(np.isfinite(transformed)):
            raise ValueError(
                "transformed control points contain non-finite values."
            )

        # Residual = predicted target - actual target.
        residuals = transformed - target

        if not np.all(np.isfinite(residuals)):
            raise ValueError(
                "computed residuals contain non-finite values."
            )

        # Euclidean error for each control point.
        point_errors = np.linalg.norm(
            residuals,
            axis=1,
        )

        if not np.all(np.isfinite(point_errors)):
            raise ValueError(
                "computed point errors contain non-finite values."
            )

        # Calculate aggregate metrics.
        rmse = float(
            np.sqrt(np.mean(np.square(point_errors)))
        )

        mean_error = float(
            np.mean(point_errors)
        )

        max_error = float(
            np.max(point_errors)
        )

        min_error = float(
            np.min(point_errors)
        )

        median_error = float(
            np.median(point_errors)
        )

        std_error = float(
            np.std(point_errors)
        )

        # Per-axis RMSE.
        rmse_x = float(
            np.sqrt(np.mean(np.square(residuals[:, 0])))
        )

        rmse_y = float(
            np.sqrt(np.mean(np.square(residuals[:, 1])))
        )

        rmse_z = float(
            np.sqrt(np.mean(np.square(residuals[:, 2])))
        )

        metrics = (
            rmse,
            mean_error,
            max_error,
            min_error,
            median_error,
            std_error,
            rmse_x,
            rmse_y,
            rmse_z,
        )

        if not all(np.isfinite(value) for value in metrics):
            raise ValueError(
                "one or more calculated validation metrics are non-finite."
            )

        # Determine pass/fail only when a tolerance was supplied.
        if self._tolerance is None:
            passed: Optional[bool] = None
        else:
            passed = bool(rmse <= self._tolerance)

        return ValidationResult(
            residuals=np.array(
                residuals,
                dtype=np.float64,
                copy=True,
            ),
            point_errors=np.array(
                point_errors,
                dtype=np.float64,
                copy=True,
            ),
            rmse=rmse,
            mean_error=mean_error,
            max_error=max_error,
            min_error=min_error,
            num_points=int(source.shape[0]),
            median_error=median_error,
            std_error=std_error,
            rmse_x=rmse_x,
            rmse_y=rmse_y,
            rmse_z=rmse_z,
            tolerance=self._tolerance,
            passed=passed,
        )

    @staticmethod
    def _validate_control_point_arrays(
        source: np.ndarray,
        target: np.ndarray,
    ) -> None:
        """
        Perform lightweight validation of control-point arrays.

        Full control-point validation is handled by ControlPoints.
        These checks protect the validator from invalid assumptions.

        Parameters
        ----------
        source : np.ndarray
            Source control points.

        target : np.ndarray
            Target control points.

        Raises
        ------
        ValueError
            If arrays have invalid dimensions, shapes, sizes,
            or contain non-finite values.
        """
        if source.ndim != 2 or source.shape[1] != 3:
            raise ValueError(
                "control_points.source must have shape (N, 3)."
            )

        if target.ndim != 2 or target.shape[1] != 3:
            raise ValueError(
                "control_points.target must have shape (N, 3)."
            )

        if source.shape != target.shape:
            raise ValueError(
                "control_points.source and control_points.target "
                "must have the same shape."
            )

        if source.shape[0] < 1:
            raise ValueError(
                "at least one control point is required for validation."
            )

        if not np.all(np.isfinite(source)):
            raise ValueError(
                "control_points.source contains non-finite values."
            )

        if not np.all(np.isfinite(target)):
            raise ValueError(
                "control_points.target contains non-finite values."
            )
