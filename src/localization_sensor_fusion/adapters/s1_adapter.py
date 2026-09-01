"""Adapter for transforming S1 perception outputs into S2 engine formats."""

from __future__ import annotations

from typing import Any
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

    def transform_for_engine(
        self, observation: S1ObservationInput
    ) -> dict[str, Any] | None:
        """Transforms validated S1 observation into COLMAP/Engine feature format.

        Returns None if frame fails quality checks.
        """
        if not self.validate_quality(observation):
            return None

        return {
            "frame_id": observation.observation_id,
            "timestamp": observation.timestamp,
            "image_path": str(observation.image),
            "camera_intrinsics": (
                observation.camera.intrinsics.model_dump()
                if observation.camera and observation.camera.intrinsics
                else None
            ),
        }