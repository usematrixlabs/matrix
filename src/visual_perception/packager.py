"""S1 Observation Packager.

Packages extracted frames, timestamps, camera calibration, quality assessments,
and keyframe flags into the canonical S1 output format (s1_output/frames/ and observations.json)
consumed by Subsystem 2 (Localization & Sensor Fusion).
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Union

from .config import S1Config
from .identifier import ObservationIdentifier
from .logger import get_logger
from .timestamp_handler import TimestampHandler
from .types import S1Output, VideoMetadataRecord


class ObservationPackager:
    """Packages and validates S1 visual perception output for downstream S2 consumption."""

    SCHEMA_VERSION: str = "1.0.0"

    def __init__(self, config: Optional[S1Config] = None):
        """Initialize the observation packager.

        Parameters:
            config (Optional[S1Config]): Subsystem configuration.
        """
        self.config = config or S1Config()
        self.logger = get_logger(self.__class__.__name__, log_level=self.config.log_level)

    @classmethod
    def get_json_schema(cls) -> Dict[str, Any]:
        """Return the standard JSON Schema definition for observations.json."""
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "MatrixS1ObservationsPackage",
            "type": "object",
            "required": [
                "schema_version",
                "subsystem",
                "total_observations",
                "observations",
            ],
            "properties": {
                "schema_version": {"type": "string"},
                "subsystem": {"type": "string"},
                "created_at": {"type": "string", "format": "date-time"},
                "video_source": {"type": ["string", "null"]},
                "total_observations": {"type": "integer"},
                "keyframe_count": {"type": "integer"},
                "keyframe_density": {"type": "number"},
                "temporal_information": {"type": "object"},
                "camera": {
                    "type": "object",
                    "required": ["width", "height"],
                    "properties": {
                        "width": {"type": "integer"},
                        "height": {"type": "integer"},
                        "intrinsics": {"type": ["object", "null"]},
                        "distortion": {"type": ["object", "null"]},
                        "is_calibrated": {"type": "boolean"},
                    },
                },
                "quality_summary": {"type": "object"},
                "observations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["observation_id", "timestamp", "image", "camera", "quality", "keyframe"],
                        "properties": {
                            "observation_id": {"type": "string"},
                            "timestamp": {"type": "number"},
                            "image": {"type": "string"},
                            "camera": {
                                "type": "object",
                                "required": ["width", "height"],
                                "properties": {
                                    "width": {"type": "integer"},
                                    "height": {"type": "integer"},
                                    "intrinsics": {"type": ["object", "null"]},
                                    "distortion": {"type": ["object", "null"]},
                                },
                            },
                            "quality": {
                                "type": "object",
                                "required": ["status"],
                                "properties": {
                                    "status": {"type": "string"},
                                    "blur_score": {"type": "number"},
                                    "quality_score": {"type": "number"},
                                    "flags": {"type": "array", "items": {"type": "string"}},
                                },
                            },
                            "keyframe": {"type": "boolean"},
                        },
                    },
                },
            },
        }

    def package(
        self,
        output_dir: Union[str, Path],
        s1_output: S1Output,
        video_metadata_record: Optional[VideoMetadataRecord] = None,
    ) -> str:
        """Package S1 output into the canonical directory structure with observations.json.

        Parameters:
            output_dir (Union[str, Path]): Target output directory (e.g. 's1_output/').
            s1_output (S1Output): Computed S1 pipeline output contract.
            video_metadata_record (Optional[VideoMetadataRecord]): Video stream metadata.

        Returns:
            str: Path to the generated observations.json file.
        """
        target_dir = Path(output_dir)
        frames_dir = target_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        frames = s1_output.visual_observations.frames
        keyframes = s1_output.visual_observations.keyframes

        # Determine camera parameters block
        calib = (
            video_metadata_record.calibration
            if video_metadata_record and video_metadata_record.calibration
            else (
                video_metadata_record.camera.calibration
                if video_metadata_record and video_metadata_record.camera
                else None
            )
        )

        w = frames[0].image_width if frames else (video_metadata_record.video.width if video_metadata_record else 1920)
        h = frames[0].image_height if frames else (video_metadata_record.video.height if video_metadata_record else 1080)

        intrinsics_dict = None
        distortion_dict = None
        is_calibrated = False

        if calib and calib.is_calibrated:
            is_calibrated = True
            intrinsics_dict = {
                "fx": calib.fx,
                "fy": calib.fy,
                "cx": calib.cx,
                "cy": calib.cy,
                "camera_matrix": calib.camera_matrix,
            }
            if calib.distortion_coefficients is not None:
                distortion_dict = {
                    "coefficients": calib.distortion_coefficients,
                    "model": calib.distortion_model or "radtan",
                }

        camera_block = {
            "width": w,
            "height": h,
            "intrinsics": intrinsics_dict,
            "distortion": distortion_dict,
            "is_calibrated": is_calibrated,
        }

        # Build observations list with portable relative image paths
        observation_items: List[Dict[str, Any]] = []

        for frame in frames:
            # Ensure frame image file is in frames_dir
            source_img = Path(frame.image_path)
            target_img = frames_dir / source_img.name

            # If not already located at target destination, copy it
            if source_img.resolve() != target_img.resolve() and source_img.exists():
                shutil.copy2(source_img, target_img)

            relative_image_path = f"frames/{source_img.name}"

            # Quality block
            q_status = frame.quality.status if frame.quality else "GOOD"
            q_blur = frame.quality.blur_score if frame.quality else 0.0
            q_score = frame.quality.quality_score if frame.quality else 100.0
            q_flags = frame.quality.flags if frame.quality else []

            item = {
                "observation_id": frame.frame_id,
                "timestamp": frame.timestamp,
                "image": relative_image_path,
                "camera": {
                    "width": frame.image_width,
                    "height": frame.image_height,
                    "intrinsics": intrinsics_dict,
                    "distortion": distortion_dict,
                },
                "quality": {
                    "status": q_status,
                    "blur_score": q_blur,
                    "quality_score": q_score,
                    "flags": q_flags,
                },
                "keyframe": frame.is_keyframe,
            }
            observation_items.append(item)

        keyframe_density = (
            round(len(keyframes) / len(frames), 4) if frames else 0.0
        )

        quality_summary = s1_output.visual_observations.visual_metadata.get("quality_summary") or {
            "GOOD": sum(1 for f in frames if f.quality and f.quality.status == "GOOD"),
            "BLURRY": sum(1 for f in frames if f.quality and f.quality.status == "BLURRY"),
            "OVEREXPOSED": sum(1 for f in frames if f.quality and f.quality.status == "OVEREXPOSED"),
            "UNDEREXPOSED": sum(1 for f in frames if f.quality and f.quality.status == "UNDEREXPOSED"),
            "LOW_FEATURE": sum(1 for f in frames if f.quality and f.quality.status == "LOW_FEATURE"),
            "CORRUPTED": sum(1 for f in frames if f.quality and f.quality.status == "CORRUPTED"),
        }

        # Build root observations.json payload
        observations_payload = {
            "schema_version": self.SCHEMA_VERSION,
            "subsystem": "S1_Visual_Perception",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "video_source": self.config.video_path,
            "total_observations": len(observation_items),
            "keyframe_count": len(keyframes),
            "keyframe_density": keyframe_density,
            "temporal_information": s1_output.temporal_information,
            "camera": camera_block,
            "quality_summary": quality_summary,
            "observations": observation_items,
        }

        observations_json_path = target_dir / "observations.json"
        with open(observations_json_path, "w", encoding="utf-8") as f:
            json.dump(observations_payload, f, indent=2)

        self.logger.info(
            "Packaged %d observations (%d keyframes) into '%s'",
            len(observation_items),
            len(keyframes),
            observations_json_path,
        )
        return str(observations_json_path.resolve())

    @classmethod
    def load_package(cls, output_dir: Union[str, Path]) -> Dict[str, Any]:
        """Load and parse a packaged S1 observations bundle programmatically.

        Parameters:
            output_dir (Union[str, Path]): Path to the packaged directory containing observations.json.

        Returns:
            Dict[str, Any]: Parsed observations.json payload.

        Raises:
            FileNotFoundError: If observations.json does not exist.
            ValueError: If the package fails integrity validation.
        """
        target_dir = Path(output_dir)
        json_file = target_dir / "observations.json"
        if not json_file.exists():
            raise FileNotFoundError(f"Packaged observations file not found: {json_file}")

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Validate package integrity
        cls.validate_package(target_dir, data)
        return data

    @classmethod
    def validate_package(cls, output_dir: Union[str, Path], package_data: Optional[Dict[str, Any]] = None) -> bool:
        """Validate package file references, ID uniqueness, and timestamp monotonicity.

        Parameters:
            output_dir (Union[str, Path]): Root package directory.
            package_data (Optional[Dict[str, Any]]): Pre-loaded dictionary data.

        Returns:
            bool: True if package is strictly valid.

        Raises:
            ValueError: If integrity validation fails.
        """
        target_dir = Path(output_dir)
        if package_data is None:
            json_file = target_dir / "observations.json"
            if not json_file.exists():
                raise FileNotFoundError(f"Packaged observations file not found: {json_file}")
            with open(json_file, "r", encoding="utf-8") as f:
                package_data = json.load(f)

        observations = package_data.get("observations")
        if not isinstance(observations, list):
            raise ValueError("Invalid observations.json: 'observations' must be a list.")

        seen_ids = set()
        prev_timestamp: Optional[float] = None

        for idx, obs in enumerate(observations):
            obs_id = obs.get("observation_id")
            if not obs_id:
                raise ValueError(f"Observation at index {idx} has missing 'observation_id'.")
            if obs_id in seen_ids:
                raise ValueError(f"Duplicate observation_id detected in package: '{obs_id}'")
            seen_ids.add(obs_id)

            # Validate timestamp
            t = obs.get("timestamp")
            if t is None or not isinstance(t, (int, float)):
                raise ValueError(f"Observation '{obs_id}' has invalid timestamp: {t!r}")
            if prev_timestamp is not None and t <= prev_timestamp:
                raise ValueError(f"Observation '{obs_id}' timestamp {t} is not strictly greater than previous {prev_timestamp}")
            prev_timestamp = t

            # Validate image file existence
            rel_image = obs.get("image")
            if not rel_image:
                raise ValueError(f"Observation '{obs_id}' has missing 'image' path.")
            image_full_path = target_dir / rel_image
            if not image_full_path.exists():
                raise ValueError(f"Observation '{obs_id}' image file does not exist: '{image_full_path}'")

            # Validate camera block
            cam = obs.get("camera")
            if not cam or "width" not in cam or "height" not in cam:
                raise ValueError(f"Observation '{obs_id}' has missing or incomplete 'camera' block.")

            # Validate quality block
            q = obs.get("quality")
            if not q or "status" not in q:
                raise ValueError(f"Observation '{obs_id}' has missing 'quality' status.")

            # Validate keyframe boolean
            if "keyframe" not in obs or not isinstance(obs["keyframe"], bool):
                raise ValueError(f"Observation '{obs_id}' has missing or non-boolean 'keyframe' flag.")

        return True

