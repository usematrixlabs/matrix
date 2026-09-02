import pytest
import numpy as np
from src.localization_sensor_fusion.utils.coordinate_transform import CoordinateTransformer
from src.localization_sensor_fusion.fusion.fusion_engine import SensorFusionEngine


def test_origin_round_trip():
    """Verify that the origin converts to (0, 0, 0) ENU and converts back to origin geodetic."""
    lat0, lon0, h0 = 12.9716, 77.5946, 920.0
    transformer = CoordinateTransformer(lat0, lon0, h0)

    # Forward origin transform
    e, n, u = transformer.geodetic_to_enu(lat0, lon0, h0)
    assert pytest.approx(e, abs=1e-5) == 0.0
    assert pytest.approx(n, abs=1e-5) == 0.0
    assert pytest.approx(u, abs=1e-5) == 0.0

    # Inverse origin transform
    lat, lon, alt = transformer.enu_to_geodetic(0.0, 0.0, 0.0)
    assert pytest.approx(lat, abs=1e-7) == lat0
    assert pytest.approx(lon, abs=1e-7) == lon0
    assert pytest.approx(alt, abs=1e-4) == h0


def test_offset_enu_round_trip():
    """Verify forward and inverse ENU conversions for offset positions."""
    lat0, lon0, h0 = 12.9716, 77.5946, 920.0
    transformer = CoordinateTransformer(lat0, lon0, h0)

    test_lat, test_lon, test_alt = 12.9725, 77.5955, 935.5
    
    # Geodetic -> ENU -> Geodetic
    e, n, u = transformer.geodetic_to_enu(test_lat, test_lon, test_alt)
    res_lat, res_lon, res_alt = transformer.enu_to_geodetic(e, n, u)

    assert pytest.approx(res_lat, abs=1e-7) == test_lat
    assert pytest.approx(res_lon, abs=1e-7) == test_lon
    assert pytest.approx(res_alt, abs=1e-4) == test_alt


def test_ecef_round_trip():
    """Verify Geodetic <-> ECEF transformations."""
    lat0, lon0, h0 = 12.9716, 77.5946, 920.0
    transformer = CoordinateTransformer(lat0, lon0, h0)

    x, y, z = transformer.geodetic_to_ecef(lat0, lon0, h0)
    res_lat, res_lon, res_alt = transformer.ecef_to_geodetic(x, y, z)

    assert pytest.approx(res_lat, abs=1e-7) == lat0
    assert pytest.approx(res_lon, abs=1e-7) == lon0
    assert pytest.approx(res_alt, abs=1e-4) == h0


test_utm_zone = "EPSG:32643"  # UTM Zone 43N (covers Bengaluru)

def test_projected_utm_round_trip():
    """Verify projection to UTM and inverse conversion back to Geodetic."""
    lat0, lon0, h0 = 12.9716, 77.5946, 920.0
    transformer = CoordinateTransformer(lat0, lon0, h0)

    utm_x, utm_y, utm_z = transformer.geodetic_to_projected(lat0, lon0, h0, test_utm_zone)
    res_lat, res_lon, res_alt = transformer.projected_to_geodetic(utm_x, utm_y, utm_z, test_utm_zone)

    assert pytest.approx(res_lat, abs=1e-7) == lat0
    assert pytest.approx(res_lon, abs=1e-7) == lon0
    assert pytest.approx(res_alt, abs=1e-4) == h0


def test_invalid_input_validation():
    """Ensure invalid latitude, longitude, NaN, and corrupt CRS definitions raise clear ValueErrors."""
    transformer = CoordinateTransformer(12.0, 77.0, 100.0)

    # Latitude out of bounds
    with pytest.raises(ValueError, match="Latitude out of valid range"):
        transformer.geodetic_to_enu(95.0, 77.0, 100.0)

    # Longitude out of bounds
    with pytest.raises(ValueError, match="Longitude out of valid range"):
        transformer.geodetic_to_enu(12.0, -190.0, 100.0)

    # Non-finite / NaN inputs
    with pytest.raises(ValueError, match="Non-finite"):
        transformer.enu_to_geodetic(np.nan, 0.0, 0.0)

    # Invalid EPSG identifier
    with pytest.raises(ValueError, match="Invalid CRS definition"):
        transformer.geodetic_to_projected(12.0, 77.0, 100.0, "EPSG:9999999")


def test_fusion_engine_gps_geodetic_boundary_integration():
    """Verify that injecting CoordinateTransformer into EKF enables direct update_gps_geodetic()."""
    lat0, lon0, h0 = 12.9716, 77.5946, 920.0
    transformer = CoordinateTransformer(lat0, lon0, h0)

    engine = SensorFusionEngine(coordinate_transformer=transformer)

    # Target fix slightly offset from origin
    target_lat, target_lon, target_alt = 12.9717, 77.5947, 925.0
    expected_e, expected_n, expected_u = transformer.geodetic_to_enu(target_lat, target_lon, target_alt)

    # Execute geodetic update
    engine.update_gps_geodetic(target_lat, target_lon, target_alt)

    # EKF internal state moves toward the converted ENU measurement from prior 0.0
    fused_pos = engine.state[0:3, 0]
    assert pytest.approx(fused_pos[0], abs=0.2) == expected_e
    assert pytest.approx(fused_pos[1], abs=0.2) == expected_n
    assert pytest.approx(fused_pos[2], abs=0.2) == expected_u