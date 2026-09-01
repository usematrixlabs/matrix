"""
S3 Input Loader

Ingests S2 localization and visual observation payloads from JSON files or dictionaries
into strongly-typed in-memory representations.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..models.schema import S2Observation, S2Payload


class S2InputLoader:
    """
    Loads and parses S2 upstream payloads for 3D reconstruction.
    """

    def __init__(self, base_directory: Optional[Union[str, Path]] = None) -> None:
        """
        Initialize the S2 input loader.

        Parameters:
            base_directory: Optional base directory to resolve relative image paths.
        """
        self.base_directory = Path(base_directory) if base_directory else None

    def load_from_file(self, json_path: Union[str, Path]) -> S2Payload:
        """
        Load an S2 payload from a JSON file.

        Parameters:
            json_path: Path to the s2_output.json file.

        Returns:
            Parsed S2Payload object.

        Raises:
            FileNotFoundError: If the json file does not exist.
            ValueError: If the file is not valid JSON or fails schema initialization.
        """
        path = Path(json_path)
        if not path.is_file():
            raise FileNotFoundError(f"S2 output JSON file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Corrupt S2 JSON file ({path}): {exc}") from exc

        base_dir = self.base_directory if self.base_directory is not None else path.parent
        return self.load_from_dict(data, base_directory=base_dir)

    def load_from_dict(
        self,
        data: Dict[str, Any],
        base_directory: Optional[Union[str, Path]] = None,
    ) -> S2Payload:
        """
        Load an S2 payload from an in-memory dictionary.

        Parameters:
            data: Dictionary representing the S2 JSON payload.
            base_directory: Optional base directory to resolve relative image paths.

        Returns:
            Parsed S2Payload object.

        Raises:
            TypeError: If data is not a dictionary.
            ValueError: If required fields are missing or invalid.
        """
        if not isinstance(data, dict):
            raise TypeError(f"Payload must be a dictionary, got {type(data).__name__}")

        if "observations" not in data:
            raise ValueError("Payload missing mandatory 'observations' list.")

        base_dir = Path(base_directory) if base_directory else self.base_directory

        payload = S2Payload.from_dict(data)

        # Resolve relative image paths if base_dir is known
        if base_dir is not None:
            for obs in payload.observations:
                img_path = Path(obs.image_path)
                if not img_path.is_absolute():
                    obs.image_path = str(base_dir / img_path)

        return payload

    @staticmethod
    def extract_feature_tracks(observations: List[S2Observation]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Organize 2D feature observations into multi-view feature tracks.

        Parameters:
            observations: List of S2Observation objects.

        Returns:
            Mapping of track_id -> list of observation dicts:
            {
                "trk_00042": [
                    {
                        "observation_id": "frame_001",
                        "feature_id": "feat_01",
                        "xy": [842.5, 412.3],
                        "rgb": [128, 140, 95],
                        "pose": CameraPose,
                        "camera": CameraIntrinsics,
                        "timestamp": 12.34
                    },
                    ...
                ]
            }
        """
        tracks: Dict[str, List[Dict[str, Any]]] = {}

        for obs in observations:
            for feat in obs.features:
                if not feat.track_id:
                    continue
                if feat.track_id not in tracks:
                    tracks[feat.track_id] = []

                tracks[feat.track_id].append({
                    "observation_id": obs.observation_id,
                    "feature_id": feat.feature_id,
                    "xy": feat.xy,
                    "rgb": feat.rgb,
                    "pose": obs.pose,
                    "camera": obs.camera,
                    "timestamp": obs.timestamp,
                })

        return tracks

