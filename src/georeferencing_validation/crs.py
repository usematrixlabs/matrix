"""
georeferencing.crs

S4 — Georeferencing & Validation subsystem
Component 3: Coordinate Reference System

This module represents and validates coordinate-reference metadata used
by the S4 georeferencing pipeline.

This module does NOT:
    - Transform coordinates.
    - Convert between CRS definitions.
    - Perform Helmert transformations.
    - Calculate validation metrics.
"""

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Optional


@dataclass
class CoordinateReference:
    """Representation of a 3D coordinate reference system.

    Attributes:
        name:
            Optional human-readable name of the coordinate reference.

        epsg:
            Optional positive EPSG identifier. The value is stored as
            metadata only; this class does not query an EPSG registry.

        units:
            Optional coordinate-unit description such as ``"meter"`` or
            ``"degree"``.

        dimension:
            Coordinate dimensionality. S4 currently requires 3D
            coordinates.

        description:
            Optional human-readable description.

        metadata:
            Optional dictionary containing additional CRS-related
            information.

    Notes:
        This class describes CRS metadata only. It does not perform
        coordinate or unit transformations.
    """

    name: Optional[str] = None
    epsg: Optional[int] = None
    units: Optional[str] = None
    dimension: int = 3
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = field(default=None)

    # S4 operates on 3D coordinates.
    REQUIRED_DIMENSION: ClassVar[int] = 3

    def __post_init__(self) -> None:
        """Validate and normalize CRS metadata."""

        self._validate_name()
        self._validate_epsg()
        self._validate_units()
        self._validate_dimension()
        self._validate_description()
        self._validate_metadata()

    def _validate_name(self) -> None:
        """Validate the optional CRS name."""

        if self.name is None:
            return

        if not isinstance(self.name, str):
            raise TypeError(
                "name must be a string when provided."
            )

        if not self.name.strip():
            raise ValueError(
                "name must be a non-empty string when provided."
            )

    def _validate_epsg(self) -> None:
        """Validate the optional EPSG identifier."""

        if self.epsg is None:
            return

        if isinstance(self.epsg, bool) or not isinstance(
            self.epsg, int
        ):
            raise TypeError(
                "epsg must be an integer when provided."
            )

        if self.epsg <= 0:
            raise ValueError(
                "epsg must be a positive integer."
            )

    def _validate_units(self) -> None:
        """Validate the optional coordinate-unit description."""

        if self.units is None:
            return

        if not isinstance(self.units, str):
            raise TypeError(
                "units must be a string when provided."
            )

        if not self.units.strip():
            raise ValueError(
                "units must be a non-empty string when provided."
            )

    def _validate_dimension(self) -> None:
        """Validate the coordinate dimensionality."""

        if isinstance(self.dimension, bool) or not isinstance(
            self.dimension, int
        ):
            raise TypeError(
                "dimension must be an integer."
            )

        if self.dimension != self.REQUIRED_DIMENSION:
            raise ValueError(
                f"dimension must be {self.REQUIRED_DIMENSION} "
                f"for S4. Received dimension={self.dimension}."
            )

    def _validate_description(self) -> None:
        """Validate the optional CRS description."""

        if self.description is not None and not isinstance(
            self.description,
            str,
        ):
            raise TypeError(
                "description must be a string when provided."
            )

    def _validate_metadata(self) -> None:
        """Validate and safely copy CRS metadata."""

        if self.metadata is None:
            self.metadata = {}
            return

        if not isinstance(self.metadata, dict):
            raise TypeError(
                "metadata must be a dictionary when provided."
            )

        self.metadata = dict(self.metadata)

    @property
    def is_3d(self) -> bool:
        """Return whether this coordinate reference is 3-dimensional."""

        return self.dimension == self.REQUIRED_DIMENSION

    def compatible_with(
        self,
        other: "CoordinateReference",
        require_units_match: bool = False,
    ) -> bool:
        """Check basic compatibility with another coordinate reference.

        This method performs only metadata-level compatibility checks.
        It does not determine whether a mathematical transformation
        exists between the two coordinate systems.

        Compatibility rules:
            - Both references must be 3D.
            - If both EPSG codes are provided, they must match.
            - If ``require_units_match`` is True, both unit values must
              be provided and equal.
            - Missing EPSG information does not automatically imply
              compatibility.

        Args:
            other:
                Another ``CoordinateReference`` instance.

            require_units_match:
                Whether explicit unit equality is required.

        Returns:
            True if the references satisfy the compatibility rules.

        Raises:
            TypeError:
                If ``other`` is not a ``CoordinateReference``.
        """

        if not isinstance(other, CoordinateReference):
            raise TypeError(
                "other must be a CoordinateReference instance."
            )

        if not self.is_3d or not other.is_3d:
            return False

        # If both EPSG codes are known, they must match.
        if (
            self.epsg is not None
            and other.epsg is not None
            and self.epsg != other.epsg
        ):
            return False

        if require_units_match:
            if self.units is None or other.units is None:
                return False

            if self.units != other.units:
                return False

        return True

    def __repr__(self) -> str:
        """Return a concise representation of the CRS."""

        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"epsg={self.epsg!r}, "
            f"units={self.units!r}, "
            f"dimension={self.dimension})"
        )