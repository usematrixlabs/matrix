"""Trajectory Estimator & Smoothing Engine for UAV Localization."""

import numpy as np
from typing import List
from src.localization_sensor_fusion.schemas.contracts import (
    S2ObservationOutput,
    Position,
)


class TrajectorySmoother:
    """Applies moving average smoothing and linear interpolation across observation trajectories."""

    def __init__(self, window_size: int = 3):
        self.window_size = window_size

    def smooth_trajectory(
        self, observations: List[S2ObservationOutput]
    ) -> List[S2ObservationOutput]:
        """Applies a centered moving average filter to x, y, z positions."""
        if len(observations) < self.window_size:
            return observations

        positions = np.array(
            [[obs.pose.position.x, obs.pose.position.y, obs.pose.position.z] for obs in observations]
        )

        smoothed_positions = np.copy(positions)
        half_window = self.window_size // 2

        for i in range(half_window, len(positions) - half_window):
            window = positions[i - half_window : i + half_window + 1]
            smoothed_positions[i] = np.mean(window, axis=0)

        for i, obs in enumerate(observations):
            obs.pose.position = Position(
                x=float(smoothed_positions[i, 0]),
                y=float(smoothed_positions[i, 1]),
                z=float(smoothed_positions[i, 2]),
            )

        return observations