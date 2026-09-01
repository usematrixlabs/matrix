"""S1 Input Adapter implementation."""

from typing import Any, Dict, Optional
from src.localization_sensor_fusion.schemas.contracts import S1ObservationInput


class S1AdapterValidationError(Exception):
    """Custom exception raised for validation errors in S1InputAdapter."""
    pass


class S1InputAdapter:
    def __init__(self, min_blur_score: float = 0.5):
        self.min_blur_score = min_blur_score

    def parse_observation(self, payload: Dict[str, Any]) -> S1ObservationInput:
        """Parses raw payload dictionary into a validated S1ObservationInput contract."""
        try:
            return S1ObservationInput(**payload)
        except Exception as e:
            raise S1AdapterValidationError(f"Invalid observation payload: {e}") from e

    def transform_for_engine(
        self, raw_input: S1ObservationInput | Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Transforms S1 input into the dictionary structure expected by COLMAP engine.
        
        Returns None if quality metrics (e.g., blur score) fail validation thresholds.
        """
        # Safely extract dictionary representation if input is a Pydantic model
        if isinstance(raw_input, S1ObservationInput):
            data = raw_input.model_dump()
        elif isinstance(raw_input, dict):
            data = raw_input
        else:
            return None

        # Validate quality threshold
        quality = data.get("quality") or {}
        if isinstance(quality, dict):
            blur_score = quality.get("blur_score", 1.0)
        else:
            blur_score = getattr(quality, "blur_score", 1.0)

        if blur_score < self.min_blur_score:
            return None

        # Extract features dictionary safely
        features = data.get("features") or {}
        if isinstance(features, dict):
            keypoints_count = features.get("keypoints_count", 0)
        else:
            keypoints_count = getattr(features, "keypoints_count", 0)

        return {
            "frame_id": data.get("observation_id"),
            "image_path": data.get("image"),
            "keypoints_count": keypoints_count,
            "timestamp": data.get("timestamp"),
        }