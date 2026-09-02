"""
georeferencing.helmert

S4 — Georeferencing & Validation subsystem
Component 4: 3D Similarity / Helmert Transformation

This module estimates and applies a 3D 7-parameter similarity
(Helmert) transformation.

Transformation convention:

    target = scale * rotation @ source + translation

The seven parameters are:

    - 3 translations: Tx, Ty, Tz
    - 3 rotations: Rx, Ry, Rz
    - 1 uniform scale

The transformation is estimated using an SVD-based Umeyama method.

This module does NOT:
    - Perform CRS conversion.
    - Calculate validation metrics.
    - Perform full RANSAC search.
    - Handle point-cloud file I/O.

Robust outlier rejection is intentionally limited to a conservative residual-based
filtering step during fit estimation so that a small number of bad control points
cannot dominate the Helmert solution.
"""

from dataclasses import dataclass
from itertools import combinations
from typing import ClassVar, Dict, Optional, Tuple

import numpy as np

from .control_points import ControlPoints


@dataclass
class HelmertTransform:
    """Represent a 3D 7-parameter similarity transformation.

    The transformation follows:

        target = scale * rotation @ source + translation

    Attributes:
        rotation:
            3x3 proper rotation matrix.

        scale:
            Positive uniform scale factor.

        translation:
            Translation vector with shape (3,).

        inlier_mask:
            Optional boolean mask indicating which input control points were
            kept after robust outlier rejection. When absent, all points were
            treated as inliers.

    Notes:
        The rotation matrix is authoritative. Euler rotation angles are
        only derived for reporting through :meth:`rotation_angles`.
    """

    rotation: np.ndarray
    scale: float
    translation: np.ndarray
    inlier_mask: Optional[np.ndarray] = None

    # Numerical validation tolerances.
    ORTHOGONALITY_TOL: ClassVar[float] = 1e-8
    DETERMINANT_TOL: ClassVar[float] = 1e-8
    NUMERICAL_TOL: ClassVar[float] = 1e-12

    def __post_init__(self) -> None:
        """Validate and normalize transformation parameters."""

        # Convert rotation to float64 NumPy array.
        try:
            self.rotation = np.asarray(
                self.rotation,
                dtype=np.float64,
            )
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "rotation must be convertible to a NumPy float64 array."
            ) from exc

        if self.rotation.shape != (3, 3):
            raise ValueError(
                "rotation must have shape (3, 3); "
                f"got {self.rotation.shape}."
            )

        if not np.all(np.isfinite(self.rotation)):
            raise ValueError(
                "rotation matrix contains non-finite values."
            )

        # Check R^T R ≈ I.
        orthogonality = self.rotation.T @ self.rotation
        identity = np.eye(3, dtype=np.float64)

        if not np.allclose(
            orthogonality,
            identity,
            atol=self.ORTHOGONALITY_TOL,
            rtol=0.0,
        ):
            raise ValueError(
                "rotation matrix is not orthogonal within tolerance."
            )

        # Check det(R) ≈ +1.
        determinant = float(np.linalg.det(self.rotation))

        if (
            not np.isfinite(determinant)
            or abs(determinant - 1.0) > self.DETERMINANT_TOL
        ):
            raise ValueError(
                "rotation matrix must be a proper rotation "
                f"with determinant +1; got det={determinant}."
            )

        # Validate scale.
        try:
            self.scale = float(self.scale)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "scale must be a numeric value."
            ) from exc

        if not np.isfinite(self.scale):
            raise ValueError(
                "scale must be finite."
            )

        if self.scale <= 0.0:
            raise ValueError(
                "scale must be strictly positive."
            )

        # Convert translation to float64 NumPy array.
        try:
            self.translation = np.asarray(
                self.translation,
                dtype=np.float64,
            )
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "translation must be convertible to a NumPy float64 array."
            ) from exc

        if self.translation.shape != (3,):
            raise ValueError(
                "translation must have shape (3,); "
                f"got {self.translation.shape}."
            )

        if not np.all(np.isfinite(self.translation)):
            raise ValueError(
                "translation contains non-finite values."
            )

        self.rotation = np.ascontiguousarray(
            self.rotation,
            dtype=np.float64,
        )

        self.translation = np.ascontiguousarray(
            self.translation,
            dtype=np.float64,
        )

        if self.inlier_mask is not None:
            inlier_mask = np.asarray(self.inlier_mask, dtype=bool)
            if inlier_mask.ndim != 1:
                raise ValueError(
                    "inlier_mask must be a 1D boolean array when provided."
                )
            self.inlier_mask = np.ascontiguousarray(inlier_mask, dtype=bool)

    @classmethod
    def from_control_points(
        cls,
        control_points: ControlPoints,
        max_iterations: int = 10,
        outlier_threshold: float = 3.0,
    ) -> "HelmertTransform":
        """Estimate a Helmert transformation from control points.

        The transformation maps source coordinates to target coordinates:

            target = scale * rotation @ source + translation

        The Umeyama SVD-based method is used to estimate the rotation,
        scale, and translation. A simple robust pass iteratively removes
        control-point outliers whose residual magnitudes are inconsistent with
        the median residual distribution.

        Args:
            control_points:
                Validated source-target control-point correspondences.
            max_iterations:
                Maximum number of robust-refinement iterations.
            outlier_threshold:
                Robust z-score threshold used to reject points with unusually
                large residual norms.

        Returns:
            An estimated :class:`HelmertTransform`.

        Raises:
            TypeError:
                If ``control_points`` is not a ``ControlPoints`` instance.

            ValueError:
                If the transformation cannot be estimated because of
                numerical degeneracy or invalid parameters.
        """

        if not isinstance(control_points, ControlPoints):
            raise TypeError(
                "control_points must be a ControlPoints instance."
            )

        if not isinstance(max_iterations, int) or max_iterations <= 0:
            raise ValueError("max_iterations must be a positive integer.")

        try:
            threshold = float(outlier_threshold)
        except (TypeError, ValueError) as exc:
            raise TypeError("outlier_threshold must be numeric.") from exc

        if not np.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("outlier_threshold must be finite and positive.")

        source = np.asarray(
            control_points.source_array,
            dtype=np.float64,
        )

        target = np.asarray(
            control_points.target_array,
            dtype=np.float64,
        )

        if source.shape != target.shape:
            raise ValueError(
                "source and target control-point arrays "
                "must have the same shape."
            )

        if source.ndim != 2 or source.shape[1] != 3:
            raise ValueError(
                "source control points must have shape (N, 3)."
            )

        if not np.all(np.isfinite(source)):
            raise ValueError(
                "source control points contain non-finite values."
            )

        if not np.all(np.isfinite(target)):
            raise ValueError(
                "target control points contain non-finite values."
            )

        n_points = source.shape[0]

        if n_points < ControlPoints.MIN_POINTS:
            raise ValueError(
                f"At least {ControlPoints.MIN_POINTS} control points "
                f"are required; got {n_points}."
            )

        best_inlier_mask = np.ones(n_points, dtype=bool)
        best_transform = cls._estimate_transform(source, target)
        best_score = float("inf")

        candidate_indices = list(combinations(np.arange(n_points, dtype=int), ControlPoints.MIN_POINTS))
        if len(candidate_indices) > 256:
            candidate_indices = candidate_indices[:256]

        for subset in candidate_indices:
            subset = np.asarray(subset, dtype=int)
            transform = cls._estimate_transform(
                source[subset],
                target[subset],
            )
            predicted = transform.transform_points(source)
            residuals = predicted - target
            residual_norms = np.linalg.norm(residuals, axis=1)
            median_norm = float(np.median(residual_norms))
            mad = float(np.median(np.abs(residual_norms - median_norm)))
            robust_scale = 1.4826 * mad if mad > 1e-12 else 1e-6
            if not np.isfinite(robust_scale) or robust_scale <= 0.0:
                robust_scale = 1e-6

            threshold_distance = median_norm + threshold * robust_scale
            inlier_mask = residual_norms <= threshold_distance
            score = float(np.median(residual_norms)) + 1e-6 * float(np.sum(residual_norms))

            if score < best_score:
                best_score = score
                best_transform = transform
                best_inlier_mask = inlier_mask

        valid_indices = np.flatnonzero(best_inlier_mask)
        if valid_indices.size < ControlPoints.MIN_POINTS:
            raise ValueError(
                "Outlier rejection eliminated too many control points; "
                "at least three inliers are required for a 3D Helmert fit."
            )

        final_transform = cls._estimate_transform(
            source[valid_indices],
            target[valid_indices],
        )
        refined_predicted = final_transform.transform_points(source)
        refined_residuals = refined_predicted - target
        refined_norms = np.linalg.norm(refined_residuals, axis=1)
        final_median = float(np.median(refined_norms))
        final_mad = float(np.median(np.abs(refined_norms - final_median)))
        final_robust_scale = 1.4826 * final_mad if final_mad > 1e-12 else 1e-6
        if not np.isfinite(final_robust_scale) or final_robust_scale <= 0.0:
            final_robust_scale = 1e-6

        final_inlier_mask = refined_norms <= (final_median + threshold * final_robust_scale)
        final_inlier_indices = np.flatnonzero(final_inlier_mask)

        if final_inlier_indices.size < ControlPoints.MIN_POINTS:
            raise ValueError(
                "Insufficient inlier control points remain for a valid Helmert fit."
            )

        final_transform = cls._estimate_transform(
            source[final_inlier_indices],
            target[final_inlier_indices],
        )
        final_transform.inlier_mask = np.zeros(n_points, dtype=bool)
        final_transform.inlier_mask[final_inlier_indices] = True
        return final_transform

    @staticmethod
    def _estimate_transform(
        source: np.ndarray,
        target: np.ndarray,
    ) -> "HelmertTransform":
        """Estimate a 3D similarity transform from a set of valid point pairs."""
        if source.ndim != 2 or source.shape[1] != 3:
            raise ValueError("source control points must have shape (N, 3).")
        if target.shape != source.shape:
            raise ValueError("source and target control points must match in shape.")

        n_points = source.shape[0]
        if n_points < ControlPoints.MIN_POINTS:
            raise ValueError(
                f"At least {ControlPoints.MIN_POINTS} control points are required; got {n_points}."
            )

        source_centroid = np.mean(source, axis=0)
        target_centroid = np.mean(target, axis=0)
        source_centered = source - source_centroid
        target_centered = target - target_centroid

        source_variance = float(np.sum(source_centered**2) / n_points)
        if not np.isfinite(source_variance) or source_variance <= HelmertTransform.NUMERICAL_TOL:
            raise ValueError(
                "source control points have insufficient geometric variance for transformation estimation."
            )

        covariance = (target_centered.T @ source_centered) / float(n_points)
        if not np.all(np.isfinite(covariance)):
            raise ValueError("cross-covariance matrix contains non-finite values.")

        try:
            U, singular_values, Vt = np.linalg.svd(covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError("SVD failed during Helmert transformation estimation.") from exc

        determinant_sign = np.linalg.det(U) * np.linalg.det(Vt)
        correction = np.eye(3, dtype=np.float64)
        if determinant_sign < 0.0:
            correction[-1, -1] = -1.0

        rotation = U @ correction @ Vt
        determinant = float(np.linalg.det(rotation))
        if not np.isfinite(determinant) or abs(determinant - 1.0) > 1e-6:
            raise ValueError(
                "Failed to construct a proper rotation matrix; "
                f"det(rotation)={determinant}."
            )

        scale_numerator = float(np.sum(singular_values * np.diag(correction)))
        scale = scale_numerator / source_variance
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("Estimated scale is not positive or is non-finite.")

        translation = target_centroid - scale * (rotation @ source_centroid)
        if not np.all(np.isfinite(translation)):
            raise ValueError("Estimated translation contains non-finite values.")

        return HelmertTransform(
            rotation=rotation,
            scale=scale,
            translation=translation,
        )

    def transform_points(
        self,
        points: np.ndarray,
    ) -> np.ndarray:
        """Transform an Nx3 array of source points.

        The transformation is:

            target = scale * rotation @ source + translation

        Args:
            points:
                Nx3 array-like containing source coordinates.

        Returns:
            Nx3 float64 array containing transformed coordinates.

        Raises:
            TypeError:
                If the input cannot be converted to numerical data.

            ValueError:
                If the input does not have shape (N, 3) or contains
                non-finite values.
        """

        try:
            points_array = np.asarray(
                points,
                dtype=np.float64,
            )
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "points must be array-like and convertible "
                "to a NumPy float64 array."
            ) from exc

        if points_array.ndim != 2:
            raise ValueError(
                "points must be a 2-dimensional array "
                "with shape (N, 3)."
            )

        if points_array.shape[1] != 3:
            raise ValueError(
                "points must have exactly 3 columns (X, Y, Z); "
                f"got shape={points_array.shape}."
            )

        if not np.all(np.isfinite(points_array)):
            raise ValueError(
                "points contain non-finite values."
            )

        # Row-wise equivalent of:
        #
        #     target = scale * R @ source + translation
        #
        # For an Nx3 array:
        #
        #     R @ source
        #
        # becomes:
        #
        #     source @ R.T
        transformed = (
            self.scale
            * (points_array @ self.rotation.T)
            + self.translation
        )

        return np.asarray(
            transformed,
            dtype=np.float64,
        )

    def transform_point(
        self,
        point: np.ndarray,
    ) -> np.ndarray:
        """Transform a single 3D point.

        Args:
            point:
                Array-like object containing [x, y, z].

        Returns:
            Transformed point as a float64 array of shape (3,).

        Raises:
            TypeError:
                If the point cannot be converted to numerical data.

            ValueError:
                If the point does not have shape (3,) or contains
                non-finite values.
        """

        try:
            point_array = np.asarray(
                point,
                dtype=np.float64,
            )
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "point must be array-like and convertible "
                "to a NumPy float64 array."
            ) from exc

        if point_array.shape != (3,):
            raise ValueError(
                "point must have shape (3,); "
                f"got {point_array.shape}."
            )

        if not np.all(np.isfinite(point_array)):
            raise ValueError(
                "point contains non-finite values."
            )

        return (
            self.scale
            * (self.rotation @ point_array)
            + self.translation
        )

    def rotation_angles(
        self,
    ) -> Tuple[float, float, float]:
        """Return Euler rotation angles in radians.

        The rotation convention is Z-Y-X:

            R = Rz(rz) @ Ry(ry) @ Rx(rx)

        The returned tuple is:

            (rx, ry, rz)

        corresponding to rotations about X, Y, and Z.

        These angles are provided only for reporting. The rotation
        matrix remains the authoritative representation.

        Returns:
            Tuple containing (rx, ry, rz) in radians.
        """

        rotation = self.rotation

        # For Z-Y-X decomposition:
        #
        #     R[2, 0] = -sin(ry)
        #
        # Clamp for numerical stability.
        value = float(
            np.clip(
                -rotation[2, 0],
                -1.0,
                1.0,
            )
        )

        ry = float(np.arcsin(value))

        cos_ry = float(np.cos(ry))

        if abs(cos_ry) > self.NUMERICAL_TOL:
            rx = float(
                np.arctan2(
                    rotation[2, 1],
                    rotation[2, 2],
                )
            )

            rz = float(
                np.arctan2(
                    rotation[1, 0],
                    rotation[0, 0],
                )
            )

        else:
            # Gimbal-lock case.
            #
            # Choose rz = 0 and determine rx from the
            # remaining rotation information.
            rz = 0.0

            if value > 0.0:
                # ry ≈ +pi/2
                rx = float(
                    np.arctan2(
                        rotation[0, 1],
                        rotation[0, 2],
                    )
                )
            else:
                # ry ≈ -pi/2
                rx = float(
                    np.arctan2(
                        -rotation[0, 1],
                        -rotation[0, 2],
                    )
                )

        return rx, ry, rz

    def parameters(self) -> Dict[str, float]:
        """Return the seven Helmert parameters.

        Returns:
            Dictionary containing:

                tx, ty, tz:
                    Translation parameters.

                rx, ry, rz:
                    Rotation angles in radians.

                scale:
                    Uniform scale factor.
        """

        rx, ry, rz = self.rotation_angles()

        return {
            "tx": float(self.translation[0]),
            "ty": float(self.translation[1]),
            "tz": float(self.translation[2]),
            "rx": rx,
            "ry": ry,
            "rz": rz,
            "scale": float(self.scale),
        }

    def as_homogeneous_matrix(self) -> np.ndarray:
        """Return the 4x4 homogeneous transformation matrix.

        The matrix represents:

            target = scale * rotation @ source + translation

        Returns:
            A 4x4 float64 matrix:

                [ scale * R   translation ]
                [     0            1      ]
        """

        matrix = np.eye(
            4,
            dtype=np.float64,
        )

        matrix[:3, :3] = self.scale * self.rotation
        matrix[:3, 3] = self.translation

        return matrix

    def __repr__(self) -> str:
        """Return a concise representation of the transformation."""

        return (
            f"{self.__class__.__name__}("
            f"scale={self.scale:.6g}, "
            f"translation={self.translation.tolist()})"
        )