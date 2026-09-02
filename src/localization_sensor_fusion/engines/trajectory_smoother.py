"""Offline, full-pose smoothing for localized UAV trajectories."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from src.localization_sensor_fusion.schemas.contracts import (
    CameraPose,
    Position,
    QuaternionOrientation,
    S2ObservationOutput,
)


class TrajectorySmoother:
    """Apply centered position averaging and hemisphere-safe quaternion SLERP.

    Matrix processes completed UAV video runs, so this smoother uses observations
    on both sides of a frame. It deliberately does not mutate the input
    trajectory: each output observation is a deep copy of its source item.
    """

    def __init__(
        self,
        window_size: int = 5,
        orientation_window_size: int | None = None,
        orientation_smoothing_strength: float = 1.0,
    ) -> None:
        self.window_size = self._validate_window_size(window_size, "window_size")
        orientation_size = window_size if orientation_window_size is None else orientation_window_size
        self.orientation_window_size = self._validate_window_size(
            orientation_size, "orientation_window_size"
        )
        if not 0.0 <= orientation_smoothing_strength <= 1.0:
            raise ValueError("orientation_smoothing_strength must be between 0.0 and 1.0")
        self.orientation_smoothing_strength = orientation_smoothing_strength

    @staticmethod
    def _validate_window_size(window_size: int, name: str) -> int:
        if not isinstance(window_size, int) or isinstance(window_size, bool) or window_size < 1:
            raise ValueError(f"{name} must be a positive odd integer")
        if window_size % 2 == 0:
            raise ValueError(f"{name} must be a positive odd integer")
        return window_size

    @staticmethod
    def _normalize_quaternion(quaternion: Sequence[float] | np.ndarray) -> np.ndarray:
        result = np.asarray(quaternion, dtype=np.float64).reshape(4)
        norm = float(np.linalg.norm(result))
        if norm <= 1e-12:
            raise ValueError("quaternion norm must be non-zero")
        return result / norm

    @classmethod
    def _align_quaternion_sign(cls, quaternion: np.ndarray, reference: np.ndarray) -> np.ndarray:
        aligned = cls._normalize_quaternion(quaternion)
        return -aligned if float(np.dot(aligned, reference)) < 0.0 else aligned

    @classmethod
    def _slerp(cls, start: np.ndarray, end: np.ndarray, fraction: float) -> np.ndarray:
        """Interpolate scalar-first quaternions on the shortest rotational arc."""
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("SLERP fraction must be between 0.0 and 1.0")
        first = cls._normalize_quaternion(start)
        second = cls._align_quaternion_sign(end, first)
        dot = float(np.clip(np.dot(first, second), -1.0, 1.0))
        if dot > 0.9995:
            return cls._normalize_quaternion(first + fraction * (second - first))
        angle = float(np.arccos(dot))
        sin_angle = float(np.sin(angle))
        return cls._normalize_quaternion(
            (np.sin((1.0 - fraction) * angle) / sin_angle) * first
            + (np.sin(fraction * angle) / sin_angle) * second
        )

    @staticmethod
    def _quaternion_from_orientation(orientation: QuaternionOrientation) -> np.ndarray:
        return np.array([orientation.qw, orientation.qx, orientation.qy, orientation.qz], dtype=np.float64)

    def _smooth_orientation(
        self,
        window: Sequence[S2ObservationOutput],
        center_orientation: QuaternionOrientation,
        center_timestamp: float,
    ) -> QuaternionOrientation:
        """Compute a timestamp-weighted quaternion mean via sequential SLERP."""
        center = self._normalize_quaternion(self._quaternion_from_orientation(center_orientation))
        samples: list[tuple[np.ndarray, float]] = []
        for observation in window:
            if observation.pose is None or observation.pose.orientation is None:
                continue
            quaternion = self._align_quaternion_sign(
                self._quaternion_from_orientation(observation.pose.orientation), center
            )
            # The centre frame has the largest influence; irregular frame
            # sampling remains well behaved because this uses capture time.
            weight = 1.0 / (1.0 + abs(float(observation.timestamp) - center_timestamp))
            samples.append((quaternion, weight))

        if not samples:
            return QuaternionOrientation(qw=float(center[0]), qx=float(center[1]), qy=float(center[2]), qz=float(center[3]))

        accumulated, total_weight = samples[0]
        for quaternion, weight in samples[1:]:
            fraction = weight / (total_weight + weight)
            accumulated = self._slerp(accumulated, quaternion, fraction)
            total_weight += weight
        smoothed = self._slerp(center, accumulated, self.orientation_smoothing_strength)
        return QuaternionOrientation(qw=float(smoothed[0]), qx=float(smoothed[1]), qy=float(smoothed[2]), qz=float(smoothed[3]))

    @staticmethod
    def _validate_timestamps(observations: Sequence[S2ObservationOutput]) -> None:
        timestamps = [float(observation.timestamp) for observation in observations]
        if any(right < left for left, right in zip(timestamps, timestamps[1:])):
            raise ValueError("observations must have nondecreasing timestamps")

    @staticmethod
    def _centered_window(observations: Sequence[S2ObservationOutput], index: int, window_size: int) -> Sequence[S2ObservationOutput]:
        half_window = window_size // 2
        return observations[max(0, index - half_window) : min(len(observations), index + half_window + 1)]

    def smooth_trajectory(self, observations: Sequence[S2ObservationOutput]) -> list[S2ObservationOutput]:
        """Return a centered-smoothed copy of a monotonic timestamp trajectory."""
        if not observations:
            return []
        self._validate_timestamps(observations)

        smoothed: list[S2ObservationOutput] = []
        for index, observation in enumerate(observations):
            output = observation.model_copy(deep=True)
            # Do not fabricate a pose when S2 explicitly has no localization.
            if observation.pose is None:
                smoothed.append(output)
                continue

            position_window = self._centered_window(observations, index, self.window_size)
            positions = [
                [item.pose.position.x, item.pose.position.y, item.pose.position.z]
                for item in position_window
                if item.pose is not None and item.pose.position is not None
            ]
            if not positions:
                smoothed.append(output)
                continue
            position = np.mean(np.asarray(positions, dtype=np.float64), axis=0)

            orientation_window = self._centered_window(observations, index, self.orientation_window_size)
            orientation = self._smooth_orientation(
                orientation_window,
                observation.pose.orientation,
                float(observation.timestamp),
            )
            output.pose = CameraPose(
                position=Position(x=float(position[0]), y=float(position[1]), z=float(position[2])),
                orientation=orientation,
            )
            smoothed.append(output)
        return smoothed
