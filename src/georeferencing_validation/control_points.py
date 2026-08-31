"""
georeferencing.control_points

S4 — Georeferencing & Validation subsystem
Component 2: Control Point Management

This module represents and validates corresponding 3D control points
used by the S4 georeferencing pipeline.

This module does NOT:
    - Estimate Helmert transformation parameters.
    - Transform coordinates.
    - Perform CRS conversion.
    - Calculate accuracy or validation metrics.
"""

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Optional, Sequence, Union

import numpy as np


ArrayLike3D = Union[
    np.ndarray,
    Sequence[Sequence[float]],
]


@dataclass
class ControlPoints:
    """Validated source-target 3D control-point correspondences.

    Source points belong to the local/reconstructed coordinate system.
    Target points belong to the reference/geographic coordinate system.

    Each source point corresponds to the target point at the same index.

    Attributes:
        source:
            Nx3 array-like containing source/local coordinates.

        target:
            Nx3 array-like containing target/reference coordinates.

        metadata:
            Optional dictionary containing additional control-point
            information such as point IDs or source references.

    Notes:
        Three non-collinear corresponding points are the theoretical
        minimum for determining a 3D 7-parameter similarity transformation.
        In real-world applications, additional control points are strongly
        recommended for robustness and error estimation.
    """

    source: ArrayLike3D
    target: ArrayLike3D
    metadata: Optional[Dict[str, Any]] = field(default=None)

    # Theoretical minimum number of point correspondences.
    MIN_POINTS: ClassVar[int] = 3

    def __post_init__(self) -> None:
        """Validate and normalize control-point data."""

        self.source = self._validate_and_convert(
            self.source,
            name="source",
        )

        self.target = self._validate_and_convert(
            self.target,
            name="target",
        )

        self._validate_correspondence()
        self._validate_minimum_points()

        self._check_duplicates(
            self.source,
            name="source",
        )

        self._check_duplicates(
            self.target,
            name="target",
        )

        self._check_geometric_rank(
            self.source,
            name="source",
        )

        self._check_geometric_rank(
            self.target,
            name="target",
        )

        if self.metadata is None:
            self.metadata = {}
        elif not isinstance(self.metadata, dict):
            try:
                self.metadata = dict(self.metadata)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "metadata must be a dictionary or convertible to a dictionary."
                ) from exc
        else:
            self.metadata = dict(self.metadata)

    @staticmethod
    def _validate_and_convert(
        points: ArrayLike3D,
        name: str,
    ) -> np.ndarray:
        """Validate and convert a control-point array.

        Parameters:
            points:
                Candidate Nx3 array-like point data.

            name:
                Name of the point set used in error messages.

        Returns:
            A contiguous NumPy float64 array of shape (N, 3).

        Raises:
            TypeError:
                If the input cannot be converted to numerical data.

            ValueError:
                If the input has an invalid shape or contains non-finite
                values.
        """

        try:
            array = np.asarray(
                points,
                dtype=np.float64,
            )
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"{name} control points must be numerical "
                "and convertible to a NumPy array."
            ) from exc

        if array.ndim != 2:
            raise ValueError(
                f"{name} control points must be a 2-dimensional "
                "array with shape (N, 3). "
                f"Received ndim={array.ndim}."
            )

        if array.shape[1] != 3:
            raise ValueError(
                f"{name} control points must contain exactly "
                f"3 coordinates (X, Y, Z). "
                f"Received shape={array.shape}."
            )

        if array.shape[0] == 0:
            raise ValueError(
                f"{name} control points cannot be empty."
            )

        if not np.all(np.isfinite(array)):
            raise ValueError(
                f"{name} control points must contain only "
                "finite numerical values. NaN and Inf are not allowed."
            )

        return np.ascontiguousarray(array)

    def _validate_correspondence(self) -> None:
        """Validate source-target point correspondence.

        Raises:
            ValueError:
                If source and target contain different numbers of points.
        """

        if self.source.shape[0] != self.target.shape[0]:
            raise ValueError(
                "Source and target control points must contain "
                "the same number of points. "
                f"Received source={self.source.shape[0]}, "
                f"target={self.target.shape[0]}."
            )

    def _validate_minimum_points(self) -> None:
        """Validate the minimum number of control-point correspondences.

        Raises:
            ValueError:
                If fewer than three correspondences are provided.
        """

        number_of_points = self.source.shape[0]

        if number_of_points < self.MIN_POINTS:
            raise ValueError(
                f"At least {self.MIN_POINTS} corresponding control points "
                "are required for a 3D similarity transformation. "
                f"Received {number_of_points}."
            )

    @staticmethod
    def _check_duplicates(
        points: np.ndarray,
        name: str,
    ) -> None:
        """Reject duplicate control-point coordinates.

        Parameters:
            points:
                Nx3 point array.

            name:
                Name of the point set.

        Raises:
            ValueError:
                If duplicate coordinates are detected.
        """

        unique_points = np.unique(
            points,
            axis=0,
        )

        if unique_points.shape[0] != points.shape[0]:
            raise ValueError(
                f"Duplicate points detected in {name} control points. "
                "Each control-point coordinate must be unique."
            )

    @staticmethod
    def _check_geometric_rank(
        points: np.ndarray,
        name: str,
    ) -> None:
        """Check whether control points provide sufficient geometry.

        The centered coordinate matrix must have rank >= 2.

        Rank < 2 indicates that the points are collinear or otherwise
        geometrically degenerate.

        Rank 2 is the minimum acceptable configuration for three
        non-collinear 3D points.

        Rank 3 indicates full 3D spatial spread and generally requires
        at least four non-coplanar points.

        Parameters:
            points:
                Nx3 control-point array.

            name:
                Name of the point set.

        Raises:
            ValueError:
                If the control points are geometrically degenerate.
        """

        centroid = np.mean(
            points,
            axis=0,
        )

        centered = points - centroid

        rank = np.linalg.matrix_rank(centered)

        if rank < 2:
            raise ValueError(
                f"{name} control points are geometrically degenerate "
                f"(rank={rank}). At least three non-collinear points "
                "are required for a 3D similarity transformation."
            )

    @property
    def number_of_points(self) -> int:
        """Return the number of control-point correspondences."""

        return int(self.source.shape[0])

    @property
    def source_array(self) -> np.ndarray:
        """Return normalized source control points as an ``(N, 3)`` array."""

        return self.source

    @property
    def target_array(self) -> np.ndarray:
        """Return normalized target control points as an ``(N, 3)`` array."""

        return self.target

    @property
    def source_centroid(self) -> np.ndarray:
        """Return the centroid of the source control points."""

        return np.mean(
            self.source,
            axis=0,
        )

    @property
    def target_centroid(self) -> np.ndarray:
        """Return the centroid of the target control points."""

        return np.mean(
            self.target,
            axis=0,
        )

    def __repr__(self) -> str:
        """Return a concise representation of the control points."""

        return (
            f"{self.__class__.__name__}("
            f"number_of_points={self.number_of_points}, "
            f"source_dtype={self.source.dtype}, "
            f"target_dtype={self.target.dtype})"
        )