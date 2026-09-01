"""Unit tests for the EKF Sensor Fusion Engine."""

import pytest
import numpy as np
from src.localization_sensor_fusion.fusion.fusion_engine import SensorFusionEngine
from src.localization_sensor_fusion.schemas.contracts import (
    S2ObservationOutput,
    CameraPose,
    Position,
    QuaternionOrientation,
    LocalizationQuality,
    LocalizationMeta,
)


def test_fusion_engine_initialization():
    engine = SensorFusionEngine()
    assert engine.state.shape == (6, 1)
    assert engine.covariance.shape == (6, 6)


def test_fusion_engine_step():
    engine = SensorFusionEngine()

    obs = S2ObservationOutput(
        observation_id="test_01",
        timestamp=10.0,
        image="test_01.jpg",
        pose=CameraPose(
            position=Position(x=10.0, y=20.0, z=30.0),
            orientation=QuaternionOrientation(qw=1.0, qx=0.0, qy=0.0, qz=0.0),
        ),
        localization=LocalizationMeta(
            source=["visual"],
            status="estimated",
            quality=LocalizationQuality(confidence=0.95),
        ),
    )

    fused_obs = engine.process_observation(obs)
    
    # Verify the fused observation has a pose with position
    assert fused_obs.pose is not None
    assert fused_obs.pose.position.x is not None
    assert fused_obs.pose.position.y is not None
    assert fused_obs.pose.position.z is not None

def test_dynamic_gps_covariance_weighting():
    engine = SensorFusionEngine()
    
    # High-confidence measurement (low variance)
    z_accurate = np.array([10.0, 10.0, 10.0])
    R_tight = np.diag([0.01, 0.01, 0.01])  # High GPS precision
    engine.update(z_accurate, R_custom=R_tight)
    
    # State should pull strongly toward accurate position
    assert np.isclose(engine.state[0], 10.0, atol=0.5)

    # Low-confidence measurement (high variance)
    z_noisy = np.array([100.0, 100.0, 100.0])
    R_loose = np.diag([50.0, 50.0, 50.0])  # Poor GPS HDOP
    engine.update(z_noisy, R_custom=R_loose)
    
    # State should resist moving toward the noisy measurement
    assert engine.state[0] < 30.0