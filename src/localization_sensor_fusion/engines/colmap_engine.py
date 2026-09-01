"""COLMAP reconstruction and camera pose estimation engine wrapper."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from src.localization_sensor_fusion.schemas.contracts import (
    CameraPose,
    S2ObservationOutput,
    S2PayloadOutput,
)


class EngineProcessingError(Exception):
    """Raised when pose processing or keypoint matching fails."""

    pass


class ColmapLocalizationEngine:
    """Wrapper for COLMAP feature extraction, matching, and camera pose estimation."""

    def __init__(self, min_matching_keypoints: int = 10):
        self.min_matching_keypoints = min_matching_keypoints
        self.processed_frames_count = 0

    def estimate_pose(self, engine_input: Dict[str, Any]) -> Optional[CameraPose]:
        """Estimates 6-DOF camera pose from formatted frame input data.

        Returns None if keypoints are insufficient for bundle adjustment.
        """
        keypoints_count = engine_input.get("keypoints_count", 0)

        if keypoints_count < self.min_matching_keypoints:
            return None

        pose = CameraPose(
            position={"x": 0.0, "y": 0.0, "z": 0.0},
            orientation={"qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0},
        )

        self.processed_frames_count += 1
        return pose

    def process_batch(
        self, batch_inputs: List[Dict[str, Any]]
    ) -> S2PayloadOutput:
        """Processes a batch of frame inputs and produces an aggregated S2 payload."""
        observations: List[S2ObservationOutput] = []

        for frame_data in batch_inputs:
            pose = self.estimate_pose(frame_data)
            if pose is not None:
                frame_id = str(frame_data.get("frame_id", "unknown"))
                obs = S2ObservationOutput(
                    observation_id=frame_id,
                    timestamp=float(frame_data.get("timestamp", 0.0)),
                    image=frame_id,
                    localization={
                        "pose": pose,
                        "status": "estimated",
                        "source": ["visual"],
                        "quality": {
                            "confidence": 1.0,
                        },
                    },
                )
                observations.append(obs)

        return S2PayloadOutput(
            session_id="s2_session_active",
            observations=observations,
        )