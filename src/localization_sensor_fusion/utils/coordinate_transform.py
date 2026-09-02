"""Coordinate Reference System transformation helper using pyproj."""

import pyproj
import numpy as np
from typing import Tuple


class CoordinateTransformer:
    """Handles CRS conversions between WGS84 (lat/lon/alt), ECEF, and local ENU."""

    def __init__(self, lat0: float, lon0: float, h0: float):
        self.lat0 = lat0
        self.lon0 = lon0
        self.h0 = h0

        # WGS84 Geodetic to ECEF
        self.wgs84 = pyproj.CRS("EPSG:4326")
        self.ecef = pyproj.CRS("EPSG:4978")
        self.transformer = pyproj.Transformer.from_crs(self.wgs84, self.ecef, always_xy=True)

        # Compute origin in ECEF
        self.x0, self.y0, self.z0 = self.geodetic_to_ecef(lat0, lon0, h0)

    def geodetic_to_ecef(self, lat: float, lon: float, alt: float) -> Tuple[float, float, float]:
        x, y, z = self.transformer.transform(lon, lat, alt)
        return float(x), float(y), float(z)

    def geodetic_to_enu(self, lat: float, lon: float, alt: float) -> Tuple[float, float, float]:
        x, y, z = self.geodetic_to_ecef(lat, lon, alt)
        dx, dy, dz = x - self.x0, y - self.y0, z - self.z0

        phi = np.radians(self.lat0)
        lam = np.radians(self.lon0)

        r_matrix = np.array([
            [-np.sin(lam), np.cos(lam), 0],
            [-np.sin(phi)*np.cos(lam), -np.sin(phi)*np.sin(lam), np.cos(phi)],
            [np.cos(phi)*np.cos(lam), np.cos(phi)*np.sin(lam), np.sin(phi)]
        ])

        enu = r_matrix @ np.array([dx, dy, dz])
        return float(enu[0]), float(enu[1]), float(enu[2])