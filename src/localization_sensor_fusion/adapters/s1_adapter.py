"""Adapter for transforming S1 perception outputs into S2 engine formats."""

from __future__ import annotations

from pyexpat import features
from typing import Any, Dict
from src.localization_sensor_fusion.schemas.contracts import (
    QualityStatus,
    S1ObservationInput,
)


class S1AdapterValidationError(Exception):
    """Raised when an S1 observation fails adapter validation rules."""


class S1InputAdapter:
    """Parses and normalizes raw S1 perception observations for the S2 localization engine."""

    def __init__(self, min_blur_score: float = 0.5):
        self.min_blur_score = min_blur_score

    def parse_observation(self, raw_data: dict[str, Any]) -> S1ObservationInput:
        """Validates raw dictionary payload into an S1ObservationInput model."""
        try:
            return S1ObservationInput(**raw_data)
        except Exception as e:
            raise S1AdapterValidationError(f"Failed to parse S1 input schema: {e}") from e

    def validate_quality(self, observation: S1ObservationInput) -> bool:
        """Verifies observation frame quality meets minimum localized threshold."""
        if observation.quality is None:
            return True

        # Reject explicitly bad quality statuses
        if observation.quality.status in (QualityStatus.BLURRY, QualityStatus.CORRUPTED):
            return False

        if observation.quality.blur_score is not None and observation.quality.blur_score < self.min_blur_score:
            return False

        return True

    def transform_for_engine(self, raw_input: Dict[str, Any]) -> Dict[str, Any]:
        """Transforms raw S1 input into COLMAP engine input structure."""
        features = raw_input.get("features", {})
        keypoints = features.get("keypoints", [])
    
        keypoints_count = raw_input.get("keypoints_count", len(keypoints))

        return {
            "frame_id": raw_input.get("frame_id"),
            "image_path": raw_input.get("image_path"),
            "keypoints_count": keypoints_count,
            "features": features,
            "timestamp": raw_input.get("timestamp"),
        }