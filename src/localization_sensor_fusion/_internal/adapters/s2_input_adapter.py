"""Adapter for ingesting and validating S2 level localization data."""

from __future__ import annotations

from typing import Any, Dict
from ..schemas.contracts import (
    CameraPose,
    S2ObservationOutput,
)


class S2InputAdapter:
    """Adapts raw localization/camera frame inputs into validated S2ObservationOutput contracts."""

    def adapt(self, raw_input: Dict[str, Any]) -> S2ObservationOutput:
        """Transforms raw dictionary input into a strictly validated S2ObservationOutput."""
        if "pose" not in raw_input or "timestamp" not in raw_input:
            raise ValueError("Invalid input: 'pose' and 'timestamp' are required fields.")

        frame_id = str(raw_input.get("frame_id", "unknown"))
        timestamp = float(raw_input["timestamp"])
        confidence = float(raw_input.get("confidence", 1.0))

        pose_data = raw_input["pose"]
        pose = CameraPose(
            position=pose_data.get("position", {"x": 0.0, "y": 0.0, "z": 0.0}),
            orientation=pose_data.get(
                "orientation", {"qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0}
            ),
        )

        return S2ObservationOutput(
            observation_id=frame_id,
            timestamp=timestamp,
            image=frame_id,
            localization={
                "pose": pose,
                "status": "estimated",
                "source": ["visual"],
                "quality": {
                    "confidence": confidence,
                },
            },
        )