"""S2 — Localization & Sensor Fusion Module Public API."""

from typing import List
from src.localization_sensor_fusion.fusion.fusion_engine import SensorFusionEngine
from src.localization_sensor_fusion.engines.trajectory_smoother import TrajectorySmoother
from src.localization_sensor_fusion.exporters.s2_exporter import S2Exporter
from src.localization_sensor_fusion.schemas.contracts import (
    S2ObservationOutput,
    S2PayloadOutput,
)


class Localizer:
    """High-level localization and trajectory estimation interface."""

    def __init__(self, window_size: int = 3):
        self.smoother = TrajectorySmoother(window_size=window_size)
        self.exporter = S2Exporter()

    def estimate_trajectory(
        self, observations: List[S2ObservationOutput]
    ) -> List[S2ObservationOutput]:
        """Smooths raw visual/localized observation trajectories."""
        return self.smoother.smooth_trajectory(observations)


class SensorFusion:
    """High-level EKF sensor fusion wrapper."""

    def __init__(self, process_noise: float = 1e-3, measurement_noise: float = 1e-2):
        self.engine = SensorFusionEngine(
            process_noise=process_noise, measurement_noise=measurement_noise
        )

    def fuse(self, observations: List[S2ObservationOutput]) -> List[S2ObservationOutput]:
        """Fuses visual observations with sensor state models."""
        return self.engine.fuse_sequence(observations)


__all__ = [
    "Localizer",
    "SensorFusion",
    "SensorFusionEngine",
    "TrajectorySmoother",
    "S2Exporter",
]