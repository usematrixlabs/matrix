"""Unit tests for S4 Coordinate Reference System (CRS) management."""

import numpy as np
import pytest

from src.georeferencing_validation._internal.crs import CoordinateReference, is_pyproj_available


def test_crs_initialization_and_validation():
    crs = CoordinateReference(
        name="WGS 84 / UTM zone 43N",
        epsg=32643,
        units="meters",
        frame_type="projected",
        datum="WGS84",
        projection="UTM",
    )
    assert crs.is_3d is True
    assert crs.is_local is False
    assert crs.is_geodetic is True
    assert crs.epsg == 32643
    assert crs.units == "meters"
    assert crs.metadata_dict["epsg"] == 32643


def test_crs_invalid_inputs():
    with pytest.raises(ValueError, match="dimension must be 3"):
        CoordinateReference(dimension=2)

    with pytest.raises(ValueError, match="frame_type must be one of"):
        CoordinateReference(frame_type="invalid_frame")

    with pytest.raises(ValueError, match="epsg must be a positive integer"):
        CoordinateReference(epsg=-4326)

    with pytest.raises(TypeError, match="name must be a string"):
        CoordinateReference(name=12345)


def test_crs_factories():
    local_crs = CoordinateReference.local(name="MY_LOCAL_FRAME", allow_local_to_world=True)
    assert local_crs.is_local is True
    assert local_crs.allow_local_to_world is True

    wgs84 = CoordinateReference.wgs84()
    assert wgs84.epsg == 4326
    assert wgs84.frame_type == "geographic"

    utm_43n = CoordinateReference.utm(zone=43, northern=True)
    assert utm_43n.epsg == 32643
    assert utm_43n.frame_type == "projected"

    utm_21s = CoordinateReference.utm(zone=21, northern=False)
    assert utm_21s.epsg == 32721


def test_crs_compatibility():
    crs1 = CoordinateReference(name="UTM 43N", epsg=32643, units="meters")
    crs2 = CoordinateReference(name="UTM 43N (alt)", epsg=32643, units="meters")
    crs3 = CoordinateReference(name="UTM 44N", epsg=32644, units="meters")

    assert crs1.compatible_with(crs2) is True
    assert crs1.compatible_with(crs3) is False


def test_crs_transform_points_same_crs():
    crs = CoordinateReference.utm(zone=43)
    pts = np.array([
        [500000.0, 3000000.0, 100.0],
        [500100.0, 3000200.0, 150.0],
    ], dtype=np.float64)

    transformed = crs.transform_points(pts, crs)
    np.testing.assert_allclose(transformed, pts)


def test_crs_transform_points_local_error():
    local_crs = CoordinateReference.local()
    utm_crs = CoordinateReference.utm(zone=43)
    pts = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)

    with pytest.raises(ValueError, match="Direct CRS coordinate conversion between a local frame and a world frame"):
        local_crs.transform_points(pts, utm_crs)

