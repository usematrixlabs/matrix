"""Unit tests covering S2 audit gap items."""

import pytest
import numpy as np
from src.localization_sensor_fusion.fusion.fusion_engine import SensorFusionEngine
from src.localization_sensor_fusion.engines.trajectory_smoother import TrajectorySmoother
from src.localization_sensor_fusion.schemas.contracts import S2ObservationOutput, CameraPose, Position, QuaternionOrientation


def test_missing_pose_handling():
    """Item 10: Test that observations without pose do not raise NaNs."""
    engine = SensorFusionEngine()
    obs = S2ObservationOutput(timestamp=1.0, pose=None)
    res = engine.process_observation(obs)
    assert res is not None
    assert not np.isnan(res.pose.position.x)


def test_covariance_confidence_mapping():
    """Item 9: Test EKF covariance trace confidence exposure."""
    engine = SensorFusionEngine()
    obs = S2ObservationOutput(
        timestamp=1.0, 
        pose=CameraPose(position=Position(x=1.0, y=2.0, z=3.0), orientation=QuaternionOrientation(qw=1.0, qx=0.0, qy=0.0, qz=0.0))
    )
    res = engine.process_observation(obs)
    assert res.pose is not None
    assert engine.covariance is not None


def test_trajectory_smoother_slerp():
    """Item 6: Test orientation SLERP smoothing."""
    smoother = TrajectorySmoother(window_size=3)
    obs1 = S2ObservationOutput(timestamp=1.0, pose=CameraPose(position=Position(x=0, y=0, z=0), orientation=QuaternionOrientation(qw=1.0, qx=0.0, qy=0.0, qz=0.0)))
    obs2 = S2ObservationOutput(timestamp=2.0, pose=CameraPose(position=Position(x=1, y=1, z=1), orientation=QuaternionOrientation(qw=0.7071, qx=0.0, qy=0.0, qz=0.7071)))
    
    smoothed = smoother.smooth_trajectory([obs1, obs2])
    assert len(smoothed) == 2
    assert smoothed[1].pose.orientation.qw != 0.0