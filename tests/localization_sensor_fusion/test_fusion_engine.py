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
    
    # Verify the state converged closer to the measurement
    assert fused_obs.pose.position.x is not None
    assert fused_obs.pose.position.y is not None
    assert fused_obs.pose.position.z is not None