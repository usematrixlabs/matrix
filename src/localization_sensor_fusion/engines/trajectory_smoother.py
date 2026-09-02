"""Trajectory smoothing with position averaging and quaternion SLERP."""

import numpy as np
from typing import List
from scipy.spatial.transform import Slerp, Rotation as R
from src.localization_sensor_fusion.schemas.contracts import S2ObservationOutput, Position, QuaternionOrientation, CameraPose


class TrajectorySmoother:
    """Smoothes position via moving average and orientation via Quaternion SLERP."""

    def __init__(self, window_size: int = 5):
        self.window_size = max(1, window_size)

    def smooth_trajectory(self, observations: List[S2ObservationOutput]) -> List[S2ObservationOutput]:
        if not observations:
            return []

        smoothed = []
        n = len(observations)

        for i in range(n):
            start_idx = max(0, i - self.window_size + 1)
            window = observations[start_idx : i + 1]

            # 1. Smooth Position (Moving Average)
            positions = [
                [obs.pose.position.x, obs.pose.position.y, obs.pose.position.z]
                for obs in window if obs.pose and obs.pose.position
            ]
            if positions:
                avg_pos = np.mean(positions, axis=0)
                smooth_x, smooth_y, smooth_z = avg_pos[0], avg_pos[1], avg_pos[2]
            else:
                smooth_x = observations[i].pose.position.x if observations[i].pose else 0.0
                smooth_y = observations[i].pose.position.y if observations[i].pose else 0.0
                smooth_z = observations[i].pose.position.z if observations[i].pose else 0.0

            # 2. Smooth Orientation (Quaternion SLERP over window)
            orientations = [
                [obs.pose.orientation.qx, obs.pose.orientation.qy, obs.pose.orientation.qz, obs.pose.orientation.qw]
                for obs in window if obs.pose and obs.pose.orientation
            ]
            
            if len(orientations) > 1:
                times = np.linspace(0, 1, len(orientations))
                rots = R.from_quat(orientations)
                slerp = Slerp(times, rots)
                interpolated_rot = slerp(1.0)  # Evaluate at latest window step
                qx, qy, qz, qw = interpolated_rot.as_quat()
            elif len(orientations) == 1:
                qx, qy, qz, qw = orientations[0]
            else:
                qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0

            # Reconstruct Observation
            curr_obs = observations[i]
            curr_obs.pose = CameraPose(
                position=Position(x=float(smooth_x), y=float(smooth_y), z=float(smooth_z)),
                orientation=QuaternionOrientation(qw=float(qw), qx=float(qx), qy=float(qy), qz=float(qz))
            )
            smoothed.append(curr_obs)

        return smoothed