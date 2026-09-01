"""Fusion Engine Layer for ingesting S2 payload observations and computing unified pose states."""

from __future__ import annotations

from typing import List, Optional
from src.localization_sensor_fusion.schemas.contracts import (
    CameraPose,
    Position,
    S2ObservationOutput,
    S2PayloadOutput,
)


class SensorFusionEngine:
    """Ingests S2 payloads and blends visual/sensor observations into a single trajectory estimate."""

    def __init__(self, min_confidence_threshold: float = 0.5):
        self.min_confidence_threshold = min_confidence_threshold

    def _get_confidence(self, obs: S2ObservationOutput) -> float:
        """Safely extracts confidence score from observation metadata."""
        loc = getattr(obs, "localization", None)
        if loc is None:
            return 1.0
        quality = getattr(loc, "quality", None)
        if quality is None:
            return 1.0
        return float(getattr(quality, "confidence", 1.0))

    def _get_pose(self, obs: S2ObservationOutput) -> CameraPose:
        """Safely extracts CameraPose from observation."""
        if hasattr(obs, "pose") and obs.pose is not None:
            return obs.pose
        loc = getattr(obs, "localization", None)
        if loc is not None and hasattr(loc, "pose"):
            return loc.pose
        raise AttributeError("S2ObservationOutput object missing 'pose' attribute.")

    def filter_observations(
        self, observations: List[S2ObservationOutput]
    ) -> List[S2ObservationOutput]:
        """Filters out observations that fail minimum quality or confidence thresholds."""
        return [
            obs for obs in observations if self._get_confidence(obs) >= self.min_confidence_threshold
        ]

    def process_payload(self, payload: S2PayloadOutput) -> Optional[CameraPose]:
        """Processes an S2PayloadOutput and computes the fused camera pose."""
        valid_observations = self.filter_observations(payload.observations)

        if not valid_observations:
            return None

        total_weight = 0.0
        avg_x = 0.0
        avg_y = 0.0
        avg_z = 0.0

        for obs in valid_observations:
            confidence = self._get_confidence(obs)
            pose = self._get_pose(obs)

            pos = pose.position
            x_val = pos.x if hasattr(pos, "x") else pos["x"]
            y_val = pos.y if hasattr(pos, "y") else pos["y"]
            z_val = pos.z if hasattr(pos, "z") else pos["z"]

            avg_x += x_val * confidence
            avg_y += y_val * confidence
            avg_z += z_val * confidence
            total_weight += confidence

        if total_weight == 0.0:
            return None

        fused_position = Position(
            x=avg_x / total_weight,
            y=avg_y / total_weight,
            z=avg_z / total_weight,
        )

        best_obs = max(valid_observations, key=self._get_confidence)
        best_pose = self._get_pose(best_obs)

        return CameraPose(
            position=fused_position,
            orientation=best_pose.orientation,
        )