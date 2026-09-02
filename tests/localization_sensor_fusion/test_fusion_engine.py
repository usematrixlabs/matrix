import pytest
import numpy as np

from src.localization_sensor_fusion._internal.fusion.fusion_engine import SensorFusionEngine
from src.localization_sensor_fusion._internal.schemas.contracts import (
    CameraPose,
    Position,
    QuaternionOrientation,
    S2ObservationOutput,
)


def test_ekf_prediction_step():
    """Verify state prediction step updates internal state over dt."""
    engine = SensorFusionEngine()
    initial_state = engine.state.copy()

    # Apply IMU gyro rates and body acceleration during prediction step
    engine.predict(dt=0.1, gyro_rates=[0.01, -0.02, 0.05], acceleration_body_mps2=[0.1, 0.0, 9.81])

    # State should have evolved
    assert not np.array_equal(engine.state, initial_state)


def test_gps_enu_update_corrects_position():
    """Verify update_gps_enu pulls position state towards measurement."""
    engine = SensorFusionEngine()

    gps_meas = [10.0, -5.0, 2.0]
    R_cov = np.eye(3) * 0.1

    engine.update_gps_enu(gps_meas, covariance_enu_m2=R_cov)

    pos_x = engine.state[0, 0] if engine.state.ndim == 2 else engine.state[0]
    pos_y = engine.state[1, 0] if engine.state.ndim == 2 else engine.state[1]
    pos_z = engine.state[2, 0] if engine.state.ndim == 2 else engine.state[2]

    assert pytest.approx(pos_x, abs=1.0) == 10.0
    assert pytest.approx(pos_y, abs=1.0) == -5.0
    assert pytest.approx(pos_z, abs=1.0) == 2.0


def test_process_observation_full_pipeline():
    """Verify full end-to-end observation processing with IMU and GPS updates."""
    engine = SensorFusionEngine()

    raw_obs = S2ObservationOutput.model_construct(
        observation_id="test_frame_0",
        timestamp=1.0,
        pose=CameraPose(
            position=Position(x=5.0, y=5.0, z=0.0),
            orientation=QuaternionOrientation(qw=1.0, qx=0.0, qy=0.0, qz=0.0),
        ),
    )

    fused_obs = engine.process_observation(
        observation=raw_obs,
        gyro_rates=[0.0, 0.0, 0.1],
        acceleration_body_mps2=[0.0, 0.0, 0.0],
        gps_enu_m=[5.1, 4.9, 0.1],
        gps_std_dev=[0.2, 0.2, 0.2],
    )

    assert fused_obs is not None
    assert fused_obs.pose is not None
    assert fused_obs.pose.position is not None
    assert fused_obs.pose.orientation is not None
    assert fused_obs.localization is not None