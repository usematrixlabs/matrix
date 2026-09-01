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

    This class stores CRS metadata and the explicit frame semantics required
    by S4. It is intentionally strict about distinguishing local coordinate
    frames from geodetic / projected world frames so that S4 does not silently
    present a local similarity transform as a true world-coordinate solution.

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

        frame_type:
            Explicit classification of the coordinate frame: ``"local"``,
            ``"geographic"``, or ``"projected"``. This is the critical
            safeguard against mislabeling a local reconstruction as a world
            CRS.

        datum:
            Optional geodetic datum name (for example ``"WGS84"``).

        projection:
            Optional map projection name (for example ``"UTM"``).

        allow_local_to_world:
            Whether a local source frame may be treated as a real-world
            georeferencing input. This must be explicitly enabled when a local
            coordinate system is mapped into a geographic or projected target.

        metadata:
            Optional dictionary containing additional CRS-related
            information.

    Notes:
        This class describes CRS metadata and the frame semantics required for
        safe georeferencing. It does not perform numerical coordinate
        conversion itself; that responsibility remains in a higher-level
        georeferencing implementation.
    """

    name: Optional[str] = None
    epsg: Optional[int] = None
    units: Optional[str] = None
    dimension: int = 3
    description: Optional[str] = None
    frame_type: str = "local"
    datum: Optional[str] = None
    projection: Optional[str] = None
    allow_local_to_world: bool = False
    metadata: Optional[Dict[str, Any]] = field(default=None)

    # S4 operates on 3D coordinates.
    REQUIRED_DIMENSION: ClassVar[int] = 3
    VALID_FRAME_TYPES: ClassVar[set[str]] = {"local", "geographic", "projected"}

    def __post_init__(self) -> None:
        """Validate and normalize CRS metadata."""

        self._validate_name()
        self._validate_epsg()
        self._validate_units()
        self._validate_dimension()
        self._validate_description()
        self._validate_frame_type()
        self._validate_datum()
        self._validate_projection()
        self._validate_local_to_world_policy()
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

    def _validate_frame_type(self) -> None:
        """Validate the explicit coordinate-frame classification."""

        if not isinstance(self.frame_type, str):
            raise TypeError(
                "frame_type must be a string identifying the coordinate frame."
            )

        normalized = self.frame_type.strip().lower()

        if normalized not in self.VALID_FRAME_TYPES:
            raise ValueError(
                "frame_type must be one of 'local', 'geographic', or 'projected'. "
                f"Received {self.frame_type!r}."
            )

        self.frame_type = normalized

    def _validate_datum(self) -> None:
        """Validate the optional datum if present."""

        if self.datum is None:
            return

        if not isinstance(self.datum, str):
            raise TypeError(
                "datum must be a string when provided."
            )

        if not self.datum.strip():
            raise ValueError(
                "datum must be a non-empty string when provided."
            )

    def _validate_projection(self) -> None:
        """Validate the optional projection if present."""

        if self.projection is None:
            return

        if not isinstance(self.projection, str):
            raise TypeError(
                "projection must be a string when provided."
            )

        if not self.projection.strip():
            raise ValueError(
                "projection must be a non-empty string when provided."
            )

    def _validate_local_to_world_policy(self) -> None:
        """Enforce explicit safety rules for local-to-world mapping."""

        if not isinstance(self.allow_local_to_world, bool):
            raise TypeError(
                "allow_local_to_world must be a boolean."
            )

        if self.metadata is None:
            self.metadata = {}

        self.metadata = dict(self.metadata)

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

    @property
    def is_local(self) -> bool:
        """Return True if the CRS is a local reconstruction frame."""

        return self.frame_type == "local"

    @property
    def is_geodetic(self) -> bool:
        """Return True if the CRS is geographic or projected."""

        return self.frame_type in {"geographic", "projected"}

    @property
    def metadata_dict(self) -> Dict[str, Any]:
        """Return CRS metadata as a dictionary."""

        return dict(self.metadata)

    def requires_explicit_world_transform(self) -> bool:
        """Return True when a local-to-world transform must be declared."""

        return self.is_geodetic and not self.allow_local_to_world

    def compatible_with(
        self,
        other: "CoordinateReference",
        require_units_match: bool = False,
    ) -> bool:
        """Check basic compatibility with another coordinate reference.

        This method performs only metadata-level compatibility checks.
        It does not determine whether a mathematical transformation exists
        between the two coordinate systems.

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
            f"dimension={self.dimension}, "
            f"frame_type={self.frame_type!r}, "
            f"datum={self.datum!r}, "
            f"projection={self.projection!r})"
        )
