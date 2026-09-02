"""
georeferencing.crs

S4 — Georeferencing & Validation subsystem
Component 3: Coordinate Reference System

Represents, validates, and manages coordinate-reference metadata and transformations
used by the S4 georeferencing pipeline. Supports geographic and projected CRS conversions
with optional PyPROJ integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Union
import numpy as np

try:
    import pyproj
except ImportError:
    pyproj = None


def is_pyproj_available() -> bool:
    """Return True if pyproj is installed and available in the runtime environment."""
    return pyproj is not None


@dataclass
class CoordinateReference:
    """Representation of a 3D coordinate reference system.

    Stores CRS metadata and explicit frame semantics required by S4.
    Distinguishes local coordinate frames from geodetic/projected world frames.

    Attributes:
        name: Optional human-readable name of the coordinate reference.
        epsg: Optional positive EPSG identifier.
        units: Optional coordinate-unit description (e.g. "meters" or "degree").
        dimension: Coordinate dimensionality (S4 requires 3D).
        description: Optional human-readable description.
        frame_type: "local", "geographic", or "projected".
        datum: Optional geodetic datum name (e.g. "WGS84").
        projection: Optional map projection name (e.g. "UTM").
        allow_local_to_world: Explicit permission to map a local frame to a world frame.
        metadata: Optional dictionary containing custom metadata.
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
        if self.name is not None and not isinstance(self.name, str):
            raise TypeError("name must be a string when provided.")

    def _validate_epsg(self) -> None:
        if self.epsg is not None:
            if not isinstance(self.epsg, int) or isinstance(self.epsg, bool):
                raise TypeError("epsg must be an integer when provided.")
            if self.epsg <= 0:
                raise ValueError("epsg must be a positive integer.")

    def _validate_units(self) -> None:
        if self.units is not None and not isinstance(self.units, str):
            raise TypeError("units must be a string when provided.")

    def _validate_dimension(self) -> None:
        if not isinstance(self.dimension, int) or isinstance(self.dimension, bool):
            raise TypeError("dimension must be an integer.")
        if self.dimension != self.REQUIRED_DIMENSION:
            raise ValueError(f"dimension must be {self.REQUIRED_DIMENSION} for 3D georeferencing.")

    def _validate_description(self) -> None:
        if self.description is not None and not isinstance(self.description, str):
            raise TypeError("description must be a string when provided.")

    def _validate_frame_type(self) -> None:
        if not isinstance(self.frame_type, str):
            raise TypeError("frame_type must be a string.")
        normalized = self.frame_type.strip().lower()
        if normalized not in self.VALID_FRAME_TYPES:
            raise ValueError(
                f"frame_type must be one of {sorted(self.VALID_FRAME_TYPES)}, got {self.frame_type!r}"
            )
        self.frame_type = normalized

    def _validate_datum(self) -> None:
        if self.datum is not None and not isinstance(self.datum, str):
            raise TypeError("datum must be a string when provided.")

    def _validate_projection(self) -> None:
        if self.projection is not None and not isinstance(self.projection, str):
            raise TypeError("projection must be a string when provided.")

    def _validate_local_to_world_policy(self) -> None:
        if not isinstance(self.allow_local_to_world, bool):
            raise TypeError("allow_local_to_world must be a boolean.")

    def _validate_metadata(self) -> None:
        if self.metadata is None:
            self.metadata = {}
        elif not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary when provided.")
        self.metadata = dict(self.metadata)

    @property
    def is_3d(self) -> bool:
        return self.dimension == self.REQUIRED_DIMENSION

    @property
    def is_local(self) -> bool:
        return self.frame_type == "local"

    @property
    def is_geodetic(self) -> bool:
        return self.frame_type in {"geographic", "projected"}

    @property
    def metadata_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "name": self.name,
            "epsg": self.epsg,
            "units": self.units,
            "dimension": self.dimension,
            "description": self.description,
            "frame_type": self.frame_type,
            "datum": self.datum,
            "projection": self.projection,
            "allow_local_to_world": self.allow_local_to_world,
        }
        if self.metadata:
            d["custom"] = dict(self.metadata)
        return d

    def requires_explicit_world_transform(self) -> bool:
        return self.is_geodetic and not self.allow_local_to_world

    def compatible_with(
        self,
        other: CoordinateReference,
        require_units_match: bool = False,
    ) -> bool:
        """Check compatibility with another coordinate reference."""
        if not isinstance(other, CoordinateReference):
            raise TypeError("other must be a CoordinateReference instance.")
        if not self.is_3d or not other.is_3d:
            return False
        if self.epsg is not None and other.epsg is not None and self.epsg != other.epsg:
            return False
        if require_units_match:
            if self.units is None or other.units is None or self.units != other.units:
                return False
        return True

    def transform_points(
        self,
        points: np.ndarray,
        target_crs: CoordinateReference,
    ) -> np.ndarray:
        """Transform an Nx3 array of points from this CRS to the target CRS."""
        if not isinstance(target_crs, CoordinateReference):
            raise TypeError("target_crs must be a CoordinateReference instance.")

        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"points must have shape (N, 3), got {pts.shape}")

        if pts.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float64)

        # 1. Identical CRS
        if self.epsg is not None and target_crs.epsg is not None and self.epsg == target_crs.epsg:
            return np.array(pts, copy=True)
        if self.is_local and target_crs.is_local:
            return np.array(pts, copy=True)

        # 2. Local <-> World safety guard
        if (self.is_local and target_crs.is_geodetic) or (self.is_geodetic and target_crs.is_local):
            raise ValueError(
                "Direct CRS coordinate conversion between a local frame and a world frame "
                "is not supported without Ground Control Points (GCPs) and a Helmert transform."
            )

        # 3. Geodetic cross-CRS conversion via PyPROJ
        if not is_pyproj_available():
            raise ImportError(
                "pyproj is required for cross-CRS geodetic coordinate transformations. "
                "Install it via 'pip install pyproj'."
            )

        if self.epsg is None or target_crs.epsg is None:
            raise ValueError("Both source and target CRS must define valid EPSG codes for transformation.")

        transformer = pyproj.Transformer.from_crs(
            f"EPSG:{self.epsg}",
            f"EPSG:{target_crs.epsg}",
            always_xy=True,
        )
        x_out, y_out, z_out = transformer.transform(pts[:, 0], pts[:, 1], pts[:, 2])
        return np.column_stack([x_out, y_out, z_out])

    @classmethod
    def from_epsg(
        cls,
        epsg: int,
        name: Optional[str] = None,
        units: str = "meters",
        frame_type: str = "projected",
        datum: Optional[str] = None,
        projection: Optional[str] = None,
        allow_local_to_world: bool = False,
    ) -> CoordinateReference:
        """Create a CoordinateReference instance from an EPSG code."""
        return cls(
            name=name or f"EPSG:{epsg}",
            epsg=epsg,
            units=units,
            frame_type=frame_type,
            datum=datum,
            projection=projection,
            allow_local_to_world=allow_local_to_world,
        )

    @classmethod
    def wgs84(cls, allow_local_to_world: bool = False) -> CoordinateReference:
        """Create a standard WGS 84 geographic 3D CRS (EPSG:4326)."""
        return cls(
            name="WGS 84 (3D)",
            epsg=4326,
            units="degrees",
            frame_type="geographic",
            datum="WGS84",
            allow_local_to_world=allow_local_to_world,
        )

    @classmethod
    def utm(
        cls,
        zone: int,
        northern: bool = True,
        allow_local_to_world: bool = False,
    ) -> CoordinateReference:
        """Create a standard WGS 84 / UTM projected 3D CRS (EPSG:32601-32660, 32701-32760)."""
        if not 1 <= zone <= 60:
            raise ValueError(f"UTM zone must be between 1 and 60, got {zone}")
        epsg = (32600 if northern else 32700) + zone
        hemisphere = "N" if northern else "S"
        return cls(
            name=f"WGS 84 / UTM zone {zone}{hemisphere}",
            epsg=epsg,
            units="meters",
            frame_type="projected",
            datum="WGS84",
            projection=f"UTM zone {zone}{hemisphere}",
            allow_local_to_world=allow_local_to_world,
        )

    @classmethod
    def local(
        cls,
        name: str = "LOCAL_RECONSTRUCTION",
        units: str = "meters",
        allow_local_to_world: bool = False,
    ) -> CoordinateReference:
        """Create a local arbitrary reconstruction CRS."""
        return cls(
            name=name,
            epsg=None,
            units=units,
            frame_type="local",
            allow_local_to_world=allow_local_to_world,
        )

    def __repr__(self) -> str:
        return (
            f"CoordinateReference(name={self.name!r}, "
            f"epsg={self.epsg!r}, "
            f"units={self.units!r}, "
            f"dimension={self.dimension}, "
            f"frame_type={self.frame_type!r}, "
            f"datum={self.datum!r}, "
            f"projection={self.projection!r})"
        )
