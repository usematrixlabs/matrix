"""Unit tests for the Trajectory Smoother Engine."""

import numpy as np
import pytest

from src.localization_sensor_fusion.engines.trajectory_smoother import TrajectorySmoother
from src.localization_sensor_fusion.schemas.contracts import (
    S2ObservationOutput,
    CameraPose,
    Position,
    QuaternionOrientation,
    LocalizationQuality,
    LocalizationMeta,
)


def _observation(
    timestamp: float,
    y: float = 0.0,
    orientation: QuaternionOrientation | None = None,
) -> S2ObservationOutput:
    return S2ObservationOutput(
        observation_id=f"frame_{timestamp}",
        timestamp=timestamp,
        image=f"frame_{timestamp}.jpg",
        pose=CameraPose(
            position=Position(x=timestamp, y=y, z=0.0),
            orientation=orientation or QuaternionOrientation(qw=1.0, qx=0.0, qy=0.0, qz=0.0),
        ),
        localization=LocalizationMeta(
            source=["visual"],
            status="estimated",
            quality=LocalizationQuality(confidence=0.95),
        ),
    )


def test_trajectory_smoother():
    smoother = TrajectorySmoother(window_size=3)

    # Create a noisy 3-frame trajectory
    obs_list = []
    y_values = [10.0, 30.0, 10.0]  # Spike in the middle frame

    for i, y_val in enumerate(y_values):
        obs = S2ObservationOutput(
            observation_id=f"frame_{i}",
            timestamp=float(i),
            image=f"frame_{i}.jpg",
            pose=CameraPose(
                position=Position(x=float(i), y=y_val, z=0.0),
                orientation=QuaternionOrientation(qw=1.0, qx=0.0, qy=0.0, qz=0.0),
            ),
            localization=LocalizationMeta(
                source=["visual"],
                status="estimated",
                quality=LocalizationQuality(confidence=0.95),
            ),
        )
        obs_list.append(obs)

    smoothed = smoother.smooth_trajectory(obs_list)

    # Middle frame y-value should be smoothed from 30.0 down to average (10+30+10)/3 = 16.67
    assert round(smoothed[1].pose.position.y, 2) == 16.67


def test_orientation_smoothing_uses_normalized_slerp():
    smoother = TrajectorySmoother(window_size=3)
    observations = [
        _observation(0.0, orientation=QuaternionOrientation(qw=1.0, qx=0.0, qy=0.0, qz=0.0)),
        _observation(1.0, orientation=QuaternionOrientation(qw=np.sqrt(0.5), qx=0.0, qy=0.0, qz=np.sqrt(0.5))),
        _observation(2.0, orientation=QuaternionOrientation(qw=1.0, qx=0.0, qy=0.0, qz=0.0)),
    ]

    smoothed = smoother.smooth_trajectory(observations)
    quaternion = smoothed[1].pose.orientation.to_numpy_scalar_first()
    assert np.isclose(np.linalg.norm(quaternion), 1.0)
    assert 0.0 < quaternion[3] < np.sqrt(0.5)


def test_quaternion_sign_equivalence_does_not_create_a_rotation_jump():
    smoother = TrajectorySmoother(window_size=3)
    identity = QuaternionOrientation(qw=1.0, qx=0.0, qy=0.0, qz=0.0)
    equivalent_identity = QuaternionOrientation(qw=-1.0, qx=0.0, qy=0.0, qz=0.0)
    smoothed = smoother.smooth_trajectory([
        _observation(0.0, orientation=identity),
        _observation(1.0, orientation=equivalent_identity),
        _observation(2.0, orientation=identity),
    ])

    assert abs(smoothed[1].pose.orientation.qw) == pytest.approx(1.0)
    assert smoothed[1].pose.orientation.qx == pytest.approx(0.0)
    assert smoothed[1].pose.orientation.qy == pytest.approx(0.0)
    assert smoothed[1].pose.orientation.qz == pytest.approx(0.0)


def test_missing_pose_is_preserved_and_inputs_are_not_mutated():
    smoother = TrajectorySmoother(window_size=3)
    with_pose = _observation(0.0, y=10.0)
    missing_pose = _observation(1.0, y=30.0).model_copy(update={"pose": None})
    final_pose = _observation(2.0, y=10.0)

    smoothed = smoother.smooth_trajectory([with_pose, missing_pose, final_pose])
    assert smoothed[1].pose is None
    assert with_pose.pose.position.y == 10.0
    assert final_pose.pose.position.y == 10.0


def test_non_monotonic_timestamps_are_rejected():
    smoother = TrajectorySmoother(window_size=3)
    with pytest.raises(ValueError, match="nondecreasing timestamps"):
        smoother.smooth_trajectory([_observation(1.0), _observation(0.0)])
