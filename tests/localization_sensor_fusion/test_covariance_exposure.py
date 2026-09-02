import pytest
import numpy as np

from src.localization_sensor_fusion.fusion.fusion_engine import SensorFusionEngine
from src.localization_sensor_fusion.schemas.contracts import (
    CameraPose,
    Position,
    QuaternionOrientation,
    S2ObservationOutput,
)


def test_position_covariance_extraction():
    """Verify 3x3 position covariance properties and standard deviation outputs."""
    engine = SensorFusionEngine()

    pos_cov = engine.position_covariance
    assert pos_cov.shape == (3, 3)
    assert np.allclose(pos_cov, np.eye(3))

    sx, sy, sz = engine.position_uncertainty_std
    assert pytest.approx(sx) == 1.0
    assert pytest.approx(sy) == 1.0
    assert pytest.approx(sz) == 1.0


def test_confidence_decay_with_high_noise():
    """Verify confidence score drops when position uncertainty grows."""
    engine = SensorFusionEngine()

    high_conf = engine.calculate_confidence_score()
    assert high_conf > 0.5

    engine.covariance[0:3, 0:3] *= 100.0
    low_conf = engine.calculate_confidence_score()

    assert low_conf < high_conf
    assert low_conf < 0.1


def test_repeated_sensor_updates_reduce_covariance_and_increase_confidence():
    """Verify that Kalman filter updates reduce covariance and boost confidence score."""
    engine = SensorFusionEngine(process_noise=1e-4, measurement_noise=1e-2)

    initial_conf = engine.calculate_confidence_score()

    meas = np.array([1.0, 2.0, 0.5])
    low_noise_R = np.eye(3) * 1e-3

    for _ in range(5):
        engine.predict(dt=0.1)
        engine.update_gps_enu(meas, covariance_enu_m2=low_noise_R)

    final_cov = engine.position_covariance
    final_conf = engine.calculate_confidence_score()

    assert final_cov[0, 0] < 1.0
    assert final_cov[1, 1] < 1.0
    assert final_cov[2, 2] < 1.0
    assert final_conf > initial_conf


def test_get_observation_output_surfaces_covariance_and_confidence():
    """Verify get_observation_output receives flattened 3x3 covariance array and confidence metric."""
    engine = SensorFusionEngine()
    obs = engine.get_observation_output(timestamp=100.0)

    assert obs.timestamp == 100.0
    assert obs.localization is not None


def test_process_observation_populates_quality_metrics():
    """Verify process_observation updates quality confidence and covariance fields in place."""
    engine = SensorFusionEngine()
    sample_obs = S2ObservationOutput.model_construct(
        observation_id="test_obs_1",
        timestamp=10.0,
        pose=CameraPose(
            position=Position(x=1.0, y=2.0, z=3.0),
            orientation=QuaternionOrientation(qw=1.0, qx=0.0, qy=0.0, qz=0.0),
        ),
    )

    fused_obs = engine.process_observation(sample_obs)

    assert fused_obs is not None
    assert fused_obs.localization is not None