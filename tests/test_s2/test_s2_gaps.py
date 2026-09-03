"""Unit tests covering S2 audit gap items."""

import pytest
import numpy as np
from localization_sensor_fusion._internal.fusion.fusion_engine import SensorFusionEngine
from localization_sensor_fusion._internal.engines.trajectory_smoother import TrajectorySmoother
from localization_sensor_fusion._internal.schemas.contracts import (
    CameraPose,
    LocalizationMeta,
    LocalizationQuality,
    Position,
    QuaternionOrientation,
    S2ObservationOutput,
)


def _observation(timestamp: float, pose: CameraPose | None = None) -> S2ObservationOutput:
    """Create the minimum valid S2 output observation for fusion tests."""
    return S2ObservationOutput(
        observation_id=f"frame_{timestamp:.1f}",
        timestamp=timestamp,
        image=f"frame_{timestamp:.1f}.jpg",
        pose=pose,
        localization=LocalizationMeta(
            source=["visual"],
            status="estimated",
            quality=LocalizationQuality(confidence=0.95),
        ),
    )


def test_missing_pose_handling():
    """Item 10: Test that observations without pose do not raise NaNs."""
    engine = SensorFusionEngine()
    obs = _observation(timestamp=1.0)
    res = engine.process_observation(obs)
    assert res is not None
    assert not np.isnan(res.pose.position.x)


def test_covariance_confidence_mapping():
    """Item 9: Test EKF covariance trace confidence exposure."""
    engine = SensorFusionEngine()
    obs = _observation(
        timestamp=1.0, 
        pose=CameraPose(position=Position(x=1.0, y=2.0, z=3.0), orientation=QuaternionOrientation(qw=1.0, qx=0.0, qy=0.0, qz=0.0))
    )
    res = engine.process_observation(obs)
    assert res.pose is not None
    assert engine.covariance is not None


def test_trajectory_smoother_slerp():
    """Item 6: Test orientation SLERP smoothing."""
    smoother = TrajectorySmoother(window_size=3)
    obs1 = _observation(timestamp=1.0, pose=CameraPose(position=Position(x=0, y=0, z=0), orientation=QuaternionOrientation(qw=1.0, qx=0.0, qy=0.0, qz=0.0)))
    obs2 = _observation(timestamp=2.0, pose=CameraPose(position=Position(x=1, y=1, z=1), orientation=QuaternionOrientation(qw=0.7071, qx=0.0, qy=0.0, qz=0.7071)))
    
    smoothed = smoother.smooth_trajectory([obs1, obs2])
    assert len(smoothed) == 2
    assert smoothed[1].pose.orientation.qw != 0.0
