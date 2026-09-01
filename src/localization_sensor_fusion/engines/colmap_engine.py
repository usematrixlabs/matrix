"""COLMAP Localization Engine implementation."""

from typing import Any, Dict, List, Optional
from src.localization_sensor_fusion.schemas.contracts import (
    CameraPose,
    Position,
    S2ObservationOutput,
)


class ColmapLocalizationEngine:
    def __init__(self, min_keypoints_threshold: int = 10):
        self.min_keypoints_threshold = min_keypoints_threshold

    def estimate_pose(self, engine_input: Dict[str, Any]) -> Optional[CameraPose]:
        """Estimates camera pose from COLMAP feature alignment."""
        keypoints_count = engine_input.get("keypoints_count", 0)

        # Check keypoint threshold
        if keypoints_count < self.min_keypoints_threshold:
            return None

        # Extract actual computed pose; return None if no real pose was solved
        raw_pose = engine_input.get("computed_pose")
        if not raw_pose:
            return None

        pos = raw_pose.get("position")
        ori = raw_pose.get("orientation")

        if not pos or not ori:
            return None

        return CameraPose(
            position=Position(**pos) if isinstance(pos, dict) else pos,
            orientation=ori,
        )

    def process_batch(self, batch_inputs: List[Dict[str, Any]]) -> List[S2ObservationOutput]:
        """Processes a batch of engine inputs into standardized S2ObservationOutput contracts."""
        results = []
        for item in batch_inputs:
            pose = self.estimate_pose(item)
            if pose is None:
                continue

            frame_id = str(item.get("frame_id", "unknown"))
            image_path = str(item.get("image_path", frame_id))
            timestamp = float(item.get("timestamp", 0.0))

            obs = S2ObservationOutput(
                observation_id=frame_id,
                timestamp=timestamp,
                image=image_path,
                pose=pose,
                localization={
                    "status": "estimated",
                    "source": ["visual"],
                    "quality": {
                        "confidence": min(1.0, item.get("keypoints_count", 0) / 100.0),
                    },
                },
            )
            results.append(obs)
        return results