"""Coordinate Reference System transformation helper using pyproj."""

import math
from typing import Optional, Tuple, Union
import numpy as np
import pyproj


class CoordinateTransformer:
    """Handles CRS conversions between WGS84 (lat/lon/alt), ECEF, local ENU, and arbitrary projected CRS."""

    EPSG_GEODETIC_3D = "EPSG:4979"  # 3D WGS84 (lon, lat, height)
    EPSG_ECEF = "EPSG:4978"         # Earth-Centered, Earth-Fixed (x, y, z)

    def __init__(self, lat0: float, lon0: float, h0: float):
        self._validate_geodetic(lat0, lon0, h0)

        self.lat0 = float(lat0)
        self.lon0 = float(lon0)
        self.h0 = float(h0)

        # WGS84 Geodetic <-> ECEF transformers
        self._geo_to_ecef_transformer = pyproj.Transformer.from_crs(
            self.EPSG_GEODETIC_3D,
            self.EPSG_ECEF,
            always_xy=True
        )
        self._ecef_to_geo_transformer = pyproj.Transformer.from_crs(
            self.EPSG_ECEF,
            self.EPSG_GEODETIC_3D,
            always_xy=True
        )

        # Compute origin in ECEF
        self.x0, self.y0, self.z0 = self.geodetic_to_ecef(self.lat0, self.lon0, self.h0)

        # Pre-compute rotation matrix and its transpose for forward/inverse ENU <-> ECEF
        phi = np.radians(self.lat0)
        lam = np.radians(self.lon0)

        sin_phi, cos_phi = np.sin(phi), np.cos(phi)
        sin_lam, cos_lam = np.sin(lam), np.cos(lam)

        self.r_matrix = np.array([
            [-sin_lam, cos_lam, 0.0],
            [-sin_phi * cos_lam, -sin_phi * sin_lam, cos_phi],
            [cos_phi * cos_lam, cos_phi * sin_lam, sin_phi]
        ], dtype=np.float64)

        self.r_matrix_t = self.r_matrix.T

    # --- Validation Helpers ---

    @staticmethod
    def _validate_geodetic(lat: float, lon: float, alt: float) -> None:
        """Validate geodetic coordinate ranges and types."""
        if not (math.isfinite(lat) and math.isfinite(lon) and math.isfinite(alt)):
            raise ValueError(f"Non-finite geodetic coordinates provided: lat={lat}, lon={lon}, alt={alt}")
        if not (-90.0 <= lat <= 90.0):
            raise ValueError(f"Latitude out of valid range [-90, 90]: {lat}")
        if not (-180.0 <= lon <= 180.0):
            raise ValueError(f"Longitude out of valid range [-180, 180]: {lon}")

    @staticmethod
    def _validate_finite_vector(name: str, *values: float) -> None:
        """Ensure input values are numeric and finite."""
        for v in values:
            if not math.isfinite(v):
                raise ValueError(f"Non-finite value in {name}: {values}")

    # --- WGS84 Geodetic <-> ECEF ---

    def geodetic_to_ecef(self, lat: float, lon: float, alt: float) -> Tuple[float, float, float]:
        self._validate_geodetic(lat, lon, alt)
        x, y, z = self._geo_to_ecef_transformer.transform(lon, lat, alt)
        self._validate_finite_vector("ECEF output", x, y, z)
        return float(x), float(y), float(z)

    def ecef_to_geodetic(self, x_m: float, y_m: float, z_m: float) -> Tuple[float, float, float]:
        self._validate_finite_vector("ECEF input", x_m, y_m, z_m)
        lon, lat, alt = self._ecef_to_geo_transformer.transform(x_m, y_m, z_m)
        self._validate_geodetic(lat, lon, alt)
        return float(lat), float(lon), float(alt)

    # --- Local ENU Transformations ---

    def geodetic_to_enu(self, lat: float, lon: float, alt: float) -> Tuple[float, float, float]:
        x, y, z = self.geodetic_to_ecef(lat, lon, alt)
        dx, dy, dz = x - self.x0, y - self.y0, z - self.z0

        enu = self.r_matrix @ np.array([dx, dy, dz], dtype=np.float64)
        return float(enu[0]), float(enu[1]), float(enu[2])

    def enu_to_ecef(self, east_m: float, north_m: float, up_m: float) -> Tuple[float, float, float]:
        self._validate_finite_vector("ENU input", east_m, north_m, up_m)
        enu = np.array([east_m, north_m, up_m], dtype=np.float64)
        d_ecef = self.r_matrix_t @ enu

        return float(d_ecef[0] + self.x0), float(d_ecef[1] + self.y0), float(d_ecef[2] + self.z0)

    def enu_to_geodetic(self, east_m: float, north_m: float, up_m: float) -> Tuple[float, float, float]:
        ecef_x, ecef_y, ecef_z = self.enu_to_ecef(east_m, north_m, up_m)
        return self.ecef_to_geodetic(ecef_x, ecef_y, ecef_z)

    # --- Generic Projected Reference Systems ---

    def transform(
        self,
        x: float,
        y: float,
        z: Optional[float],
        source_crs: Union[str, int],
        target_crs: Union[str, int],
    ) -> Tuple[float, float, Optional[float]]:
        self._validate_finite_vector("Transform input", x, y, 0.0 if z is None else z)
        try:
            src = pyproj.CRS.from_user_input(source_crs)
            tgt = pyproj.CRS.from_user_input(target_crs)
            transformer = pyproj.Transformer.from_crs(src, tgt, always_xy=True)
        except Exception as e:
            raise ValueError(f"Invalid CRS definition: source='{source_crs}', target='{target_crs}'. Error: {e}")

        if z is None:
            res_x, res_y = transformer.transform(x, y)
            res_z = None
        else:
            res_x, res_y, res_z = transformer.transform(x, y, z)
            res_z = float(res_z)

        self._validate_finite_vector("Transform output", res_x, res_y, 0.0 if res_z is None else res_z)
        return float(res_x), float(res_y), res_z

    def geodetic_to_projected(
        self,
        lat: float,
        lon: float,
        alt: float,
        target_crs: Union[str, int],
    ) -> Tuple[float, float, float]:
        self._validate_geodetic(lat, lon, alt)
        res_x, res_y, res_z = self.transform(lon, lat, alt, self.EPSG_GEODETIC_3D, target_crs)
        return res_x, res_y, res_z if res_z is not None else alt

    def projected_to_geodetic(
        self,
        x_m: float,
        y_m: float,
        z_m: float,
        source_crs: Union[str, int],
    ) -> Tuple[float, float, float]:
        lon, lat, alt = self.transform(x_m, y_m, z_m, source_crs, self.EPSG_GEODETIC_3D)
        if alt is None:
            alt = z_m
        self._validate_geodetic(lat, lon, alt)
        return lat, lon, alt