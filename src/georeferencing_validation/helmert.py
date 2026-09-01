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
    - Perform outlier rejection.
    - Perform RANSAC.
    - Handle point-cloud file I/O.
"""

from dataclasses import dataclass
from typing import ClassVar, Dict, Tuple

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

    Notes:
        The rotation matrix is authoritative. Euler rotation angles are
        only derived for reporting through :meth:`rotation_angles`.
    """

    rotation: np.ndarray
    scale: float
    translation: np.ndarray

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

    @classmethod
    def from_control_points(
        cls,
        control_points: ControlPoints,
    ) -> "HelmertTransform":
        """Estimate a Helmert transformation from control points.

        The transformation maps source coordinates to target coordinates:

            target = scale * rotation @ source + translation

        The Umeyama SVD-based method is used to estimate the rotation,
        scale, and translation.

        Args:
            control_points:
                Validated source-target control-point correspondences.

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

        # ---------------------------------------------------------
        # Step 1: Calculate centroids.
        # ---------------------------------------------------------
        source_centroid = np.mean(
            source,
            axis=0,
        )

        target_centroid = np.mean(
            target,
            axis=0,
        )

        # ---------------------------------------------------------
        # Step 2: Center the coordinates.
        # ---------------------------------------------------------
        source_centered = source - source_centroid
        target_centered = target - target_centroid

        # ---------------------------------------------------------
        # Step 3: Calculate source variance.
        #
        # Umeyama uses:
        #
        #     variance = sum(||Xc||²) / N
        # ---------------------------------------------------------
        source_variance = float(
            np.sum(source_centered**2) / n_points
        )

        if (
            not np.isfinite(source_variance)
            or source_variance <= cls.NUMERICAL_TOL
        ):
            raise ValueError(
                "source control points have insufficient "
                "geometric variance for transformation estimation."
            )

        # ---------------------------------------------------------
        # Step 4: Calculate cross-covariance matrix.
        #
        # Sigma = (Yc^T Xc) / N
        #
        # This convention estimates the transformation:
        #
        #     Y = s R X + t
        # ---------------------------------------------------------
        covariance = (
            target_centered.T @ source_centered
        ) / float(n_points)

        if not np.all(np.isfinite(covariance)):
            raise ValueError(
                "cross-covariance matrix contains non-finite values."
            )

        # ---------------------------------------------------------
        # Step 5: SVD of covariance matrix.
        # ---------------------------------------------------------
        try:
            U, singular_values, Vt = np.linalg.svd(
                covariance
            )
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "SVD failed during Helmert transformation estimation."
            ) from exc

        # ---------------------------------------------------------
        # Step 6: Prevent an unintended reflection.
        #
        # R = U D V^T
        #
        # D is chosen so det(R) = +1.
        # ---------------------------------------------------------
        determinant_sign = (
            np.linalg.det(U) * np.linalg.det(Vt)
        )

        correction = np.eye(
            3,
            dtype=np.float64,
        )

        if determinant_sign < 0.0:
            correction[-1, -1] = -1.0

        rotation = U @ correction @ Vt

        determinant = float(
            np.linalg.det(rotation)
        )

        if (
            not np.isfinite(determinant)
            or abs(determinant - 1.0) > 1e-6
        ):
            raise ValueError(
                "Failed to construct a proper rotation matrix; "
                f"det(rotation)={determinant}."
            )

        # ---------------------------------------------------------
        # Step 7: Estimate uniform scale.
        #
        #     s = trace(Sigma-related term) / variance
        #
        # For the Umeyama formulation:
        #
        #     s = sum(D_i * correction_i) / variance
        # ---------------------------------------------------------
        scale_numerator = float(
            np.sum(
                singular_values
                * np.diag(correction)
            )
        )

        scale = scale_numerator / source_variance

        if not np.isfinite(scale):
            raise ValueError(
                "Estimated scale is non-finite."
            )

        if scale <= 0.0:
            raise ValueError(
                "Estimated scale is not positive."
            )

        # ---------------------------------------------------------
        # Step 8: Estimate translation.
        #
        #     t = μY - s R μX
        # ---------------------------------------------------------
        translation = (
            target_centroid
            - scale * (rotation @ source_centroid)
        )

        if not np.all(np.isfinite(translation)):
            raise ValueError(
                "Estimated translation contains non-finite values."
            )

        return cls(
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