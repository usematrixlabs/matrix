"""Unit tests for the Trajectory Smoother Engine."""

from src.localization_sensor_fusion.engines.trajectory_smoother import TrajectorySmoother
from src.localization_sensor_fusion.schemas.contracts import (
    S2ObservationOutput,
    CameraPose,
    Position,
    QuaternionOrientation,
    LocalizationQuality,
    LocalizationMeta,
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