"""S4 Georeferencer

Transforms local 3D reconstruction into geographic coordinates.
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class ReconstructionInput:
    """Validated input data received from S3.

    Attributes:
        points: Nx3 NumPy array containing X, Y, Z coordinates.
        colors: Optional Nx3 NumPy array containing RGB values.
    """

    points: np.ndarray
    colors: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        """Validate reconstruction input after initialization."""
        self.points = self._validate_points(self.points)

        if self.colors is not None:
            self.colors = self._validate_colors(
                self.colors,
                len(self.points),
            )

    @staticmethod
    def _validate_points(points: Any) -> np.ndarray:
        """Validate and normalize a 3D point cloud.

        Parameters:
            points: Point-cloud data expected to represent an Nx3 array.

        Returns:
            A validated NumPy float64 array with shape (N, 3).

        Raises:
            TypeError: If points cannot be converted to numerical data.
            ValueError: If points have an invalid shape or contain
                non-finite values.
        """
        try:
            points_array = np.asarray(points, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "Point cloud must contain numerical values."
            ) from exc

        if points_array.ndim != 2:
            raise ValueError(
                "Point cloud must be a 2-dimensional Nx3 array."
            )

        if points_array.shape[1] != 3:
            raise ValueError(
                "Point cloud must contain exactly 3 columns "
                "(X, Y, Z)."
            )

        if points_array.shape[0] == 0:
            raise ValueError(
                "Point cloud must contain at least one point."
            )

        if not np.all(np.isfinite(points_array)):
            raise ValueError(
                "Point cloud must contain only finite numerical values."
            )

        return np.ascontiguousarray(points_array)

    @staticmethod
    def _validate_colors(
        colors: Any,
        number_of_points: int,
    ) -> np.ndarray:
        """Validate RGB color data.

        Parameters:
            colors: RGB values expected to be an Nx3 array.
            number_of_points: Number of points in the point cloud.

        Returns:
            A validated NumPy uint8 array with shape (N, 3).

        Raises:
            TypeError: If color data is not numerical.
            ValueError: If color dimensions, count, or values are invalid.
        """
        try:
            colors_array = np.asarray(colors)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "Color data must be convertible to a NumPy array."
            ) from exc

        if colors_array.ndim != 2 or colors_array.shape[1] != 3:
            raise ValueError(
                "Color data must be an Nx3 RGB array."
            )

        if colors_array.shape[0] != number_of_points:
            raise ValueError(
                "Number of color entries must match the number "
                "of points."
            )

        if not np.issubdtype(colors_array.dtype, np.number):
            raise TypeError(
                "Color data must contain numerical RGB values."
            )

        if not np.all(np.isfinite(colors_array)):
            raise ValueError(
                "Color data must contain only finite values."
            )

        if np.any(colors_array < 0) or np.any(colors_array > 255):
            raise ValueError(
                "RGB values must be in the range [0, 255]."
            )

        if np.any(colors_array != np.floor(colors_array)):
            raise ValueError(
                "RGB values must be integer-valued."
            )

        return np.ascontiguousarray(colors_array, dtype=np.uint8)


class Georeferencer:
    """Georeference the reconstructed 3D scene."""

    def __init__(
        self,
        reconstruction_data: Any,
        coordinate_reference: dict,
    ):
        """Initialize the georeferencer.

        Parameters:
            reconstruction_data:
                Local 3D reconstruction received from S3.
                It should contain or represent an Nx3 point cloud.

            coordinate_reference:
                Dictionary containing metadata describing the
                target geographic coordinate reference.

        Raises:
            TypeError: If coordinate_reference is not a dictionary.
            ValueError: If reconstruction data is invalid.
        """
        if not isinstance(coordinate_reference, dict):
            raise TypeError(
                "coordinate_reference must be a dictionary."
            )

        self.reconstruction_data = reconstruction_data
        self.coordinate_reference = coordinate_reference

        self.input_data = self._prepare_input(reconstruction_data)

    @staticmethod
    def _prepare_input(
        reconstruction_data: Any,
    ) -> ReconstructionInput:
        """Convert S3 reconstruction data into validated S4 input.

        Parameters:
            reconstruction_data:
                S3 reconstructed point-cloud data.

        Returns:
            Validated ReconstructionInput.

        Raises:
            TypeError: If the reconstruction format is unsupported.
            ValueError: If required point-cloud data is invalid.
        """
        if isinstance(reconstruction_data, ReconstructionInput):
            return reconstruction_data

        # Direct Nx3 NumPy array/list input.
        if isinstance(reconstruction_data, np.ndarray):
            return ReconstructionInput(points=reconstruction_data)

        if isinstance(reconstruction_data, (list, tuple)):
            return ReconstructionInput(points=reconstruction_data)

        # Dictionary-based S3 output.
        if isinstance(reconstruction_data, dict):
            if "points" not in reconstruction_data:
                raise ValueError(
                    "reconstruction_data must contain a 'points' field."
                )

            points = reconstruction_data["points"]
            colors = reconstruction_data.get("colors")

            return ReconstructionInput(
                points=points,
                colors=colors,
            )

        raise TypeError(
            "Unsupported reconstruction_data format. "
            "Expected ReconstructionInput, NumPy array, "
            "list, tuple, or dictionary containing 'points'."
        )

    def georeference(self) -> dict:
        """Prepare the reconstruction's geographic reference result.

        Returns:
            Dictionary containing placeholders for the future
            georeferenced point cloud, geographic reference, and metrics.

        Note:
            Helmert transformation and validation are not implemented yet.
        """
        # TODO: Implement 3D Helmert georeferencing.
        return {
            "geo_point_cloud": [],
            "geographic_reference": self.coordinate_reference,
            "metrics": {},
        }