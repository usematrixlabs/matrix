"""
georeferencing.input

S4 — Georeferencing & Validation subsystem
Component 1: Input & Data Handling

This module defines a validated representation of the reconstructed 3D
point-cloud data received from S3.

Responsibilities:
    - Represent reconstructed 3D points.
    - Validate point-cloud coordinates.
    - Validate optional RGB/color information.
    - Preserve optional metadata.

This module does NOT:
    - Perform coordinate transformations.
    - Perform CRS conversion.
    - Estimate Helmert transformation parameters.
    - Perform georeferencing.
    - Calculate validation metrics.

Those responsibilities belong to other S4 components.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np


# Supported input format for point and color data.
ArrayLike = Union[np.ndarray, Sequence[Sequence[float]]]


@dataclass
class ReconstructionInput:
    """Validated input data received from S3.

    Attributes:
        points:
            Nx3 array containing X, Y, Z coordinates of the reconstructed
            point cloud. The stored representation is NumPy float64.

        colors:
            Optional Nx3 RGB array containing one RGB value for each point.
            Values must be in the range [0, 255]. The stored representation
            is NumPy uint8.

        metadata:
            Optional dictionary containing additional metadata provided by S3.
    """

    points: ArrayLike
    colors: Optional[ArrayLike] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize the supplied input data."""

        self.points = self._validate_points(self.points)

        if self.colors is not None:
            self.colors = self._validate_colors(
                self.colors,
                self.points.shape[0],
            )

        if not isinstance(self.metadata, dict):
            raise TypeError(
                "metadata must be a dictionary."
            )

        # Make a shallow copy so external modifications do not affect
        # the stored metadata.
        self.metadata = dict(self.metadata)

    @staticmethod
    def _validate_points(points: ArrayLike) -> np.ndarray:
        """Validate and convert a 3D point cloud.

        Parameters:
            points:
                Array-like object representing an Nx3 point cloud.

        Returns:
            NumPy array of shape (N, 3) and dtype float64.

        Raises:
            TypeError:
                If the input cannot be converted to numerical data.

            ValueError:
                If the input does not have the required shape or contains
                invalid numerical values.
        """

        try:
            points_array = np.asarray(
                points,
                dtype=np.float64,
            )
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "points must be a numerical array-like object."
            ) from exc

        if points_array.ndim != 2:
            raise ValueError(
                "points must be a 2-dimensional array "
                "with shape (N, 3)."
            )

        if points_array.shape[1] != 3:
            raise ValueError(
                "points must contain exactly 3 columns "
                "(X, Y, Z)."
            )

        if points_array.shape[0] == 0:
            raise ValueError(
                "points must contain at least one point."
            )

        if not np.all(np.isfinite(points_array)):
            raise ValueError(
                "points must contain only finite numerical values. "
                "NaN and Inf are not allowed."
            )

        return np.ascontiguousarray(points_array)

    @staticmethod
    def _validate_colors(
        colors: ArrayLike,
        expected_points: int,
    ) -> np.ndarray:
        """Validate and convert RGB color information.

        Parameters:
            colors:
                Array-like Nx3 RGB data.

            expected_points:
                Number of points in the associated point cloud.

        Returns:
            NumPy array of shape (N, 3) and dtype uint8.

        Raises:
            TypeError:
                If colors cannot be interpreted as numerical data.

            ValueError:
                If the shape, number of rows, range, or values are invalid.
        """

        try:
            colors_array = np.asarray(colors)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "colors must be a numerical array-like object."
            ) from exc

        if colors_array.ndim != 2:
            raise ValueError(
                "colors must be a 2-dimensional array "
                "with shape (N, 3)."
            )

        if colors_array.shape[1] != 3:
            raise ValueError(
                "colors must contain exactly 3 columns "
                "(R, G, B)."
            )

        if colors_array.shape[0] != expected_points:
            raise ValueError(
                "colors must contain exactly one RGB value "
                "for each point. "
                f"Expected {expected_points}, "
                f"received {colors_array.shape[0]}."
            )

        try:
            colors_float = np.asarray(
                colors_array,
                dtype=np.float64,
            )
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "colors must contain numerical RGB values."
            ) from exc

        if not np.all(np.isfinite(colors_float)):
            raise ValueError(
                "colors must contain only finite numerical values. "
                "NaN and Inf are not allowed."
            )

        if np.any(colors_float < 0) or np.any(colors_float > 255):
            raise ValueError(
                "RGB values must be within the range [0, 255]."
            )

        # RGB values represent discrete color channels.
        if np.any(colors_float != np.floor(colors_float)):
            raise ValueError(
                "RGB values must be integer-valued."
            )

        return np.ascontiguousarray(
            colors_float,
            dtype=np.uint8,
        )

    @property
    def num_points(self) -> int:
        """Return the number of points in the point cloud."""
        return int(self.points.shape[0])

    @property
    def has_colors(self) -> bool:
        """Return True if RGB color information is available."""
        return self.colors is not None

    @property
    def points_array(self) -> np.ndarray:
        """Return point cloud coordinates as a NumPy float64 array."""
        return self.points

    @property
    def colors_array(self) -> Optional[np.ndarray]:
        """Return RGB color data as a NumPy uint8 array if available."""
        return self.colors

    @property
    def metadata_dict(self) -> Dict[str, Any]:
        """Return metadata as a dictionary."""
        return self.metadata