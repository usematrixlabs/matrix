"""
S3 Reconstruction Schema Models

Minimal dataclasses consumed by the S3 input loader/validator/preparer/
engine/output packager. These exist so that ``src.reconstruction`` imports
cleanly and the existing internals can construct / serialize S2 payloads
and S3 results without modification.

This module is intentionally minimal — only the attributes and methods
actually referenced by S3 internals are defined.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class S3Status(str, Enum):
    """Reconstruction status codes returned by S3."""

    SUCCESS = "success"
    WARNING = "warning"
    PARTIAL = "partial"
    FAILURE = "failure"
    INVALID_INPUT = "invalid_input"


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsic matrix parameters."""

    fx: float = 1.0
    fy: float = 1.0
    cx: float = 0.0
    cy: float = 0.0
    width: Optional[int] = None
    height: Optional[int] = None
    distortion_coefficients: Optional[List[float]] = None
    distortion_model: Optional[str] = None

    def to_matrix(self) -> np.ndarray:
        """Return the 3x3 intrinsic matrix K."""
        return np.array(
            [
                [float(self.fx), 0.0, float(self.cx)],
                [0.0, float(self.fy), float(self.cy)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


@dataclass
class CameraPose:
    """Camera pose with position and orientation."""

    position: Any = None
    orientation: Any = None
    orientation_format: str = "QUATERNION_XYZW"

    @property
    def position_array(self) -> np.ndarray:
        """Position as a 3-element float64 array."""
        if self.position is None:
            return np.zeros(3, dtype=np.float64)
        return np.asarray(self.position, dtype=np.float64).reshape(3)

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Orientation as a 3x3 rotation matrix."""
        fmt = (self.orientation_format or "").upper()
        if fmt == "ROTATION_MATRIX":
            return np.asarray(self.orientation, dtype=np.float64).reshape(3, 3)
        if fmt in ("QUATERNION_XYZW", "QUATERNION_WXYZ"):
            return self._quaternion_to_rotation(self.orientation)
        return np.eye(3, dtype=np.float64)

    @staticmethod
    def _quaternion_to_rotation(q: Any) -> np.ndarray:
        q_arr = np.asarray(q, dtype=np.float64).reshape(4)
        fmt_norm = "wxyz"
        x, y, z, w = q_arr
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    def projection_matrix(self, camera: Optional[CameraIntrinsics]) -> np.ndarray:
        """Return the 3x4 projection matrix P = K [R | t]."""
        if camera is None:
            return np.zeros((3, 4), dtype=np.float64)
        k = camera.to_matrix()
        r = self.rotation_matrix
        t = self.position_array
        rt = np.hstack([r, t.reshape(3, 1)])
        return k @ rt


@dataclass
class FeatureObservation:
    """A single 2D feature observation tied to a S2 observation."""

    feature_id: str = ""
    xy: Any = (0.0, 0.0)
    rgb: Optional[Any] = None
    track_id: Optional[str] = None
    descriptor: Optional[Any] = None
    response: Optional[float] = None


@dataclass
class LocalizationInfo:
    """Localization metadata attached to an S2 observation."""

    status: str = "estimated"
    source: List[str] = field(default_factory=lambda: ["visual"])
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class S2Observation:
    """A single S2 observation: image + camera + pose + features."""

    observation_id: str
    timestamp: float
    image_path: str
    camera: Optional[CameraIntrinsics] = None
    pose: Optional[CameraPose] = None
    features: List[FeatureObservation] = field(default_factory=list)
    localization: LocalizationInfo = field(default_factory=LocalizationInfo)


@dataclass
class S2Payload:
    """S2 → S3 payload: collection of localized observations."""

    observations: List[S2Observation] = field(default_factory=list)
    job_id: Optional[str] = None
    coordinate_frame: str = "local"
    units: str = "meters"
    schema_version: str = "1.0.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "S2Payload":
        """Construct an S2Payload from a dict (the canonical JSON shape)."""
        if not isinstance(data, dict):
            raise TypeError(f"S2Payload expects a dict, got {type(data).__name__}")

        raw_obs = data.get("observations", [])
        observations: List[S2Observation] = []
        for raw in raw_obs:
            if not isinstance(raw, dict):
                continue
            cam_raw = raw.get("camera") or raw.get("intrinsics") or {}
            camera = None
            if isinstance(cam_raw, dict):
                camera = CameraIntrinsics(
                    fx=float(cam_raw.get("fx", cam_raw.get("focal_length_x", 1.0)) or 1.0),
                    fy=float(cam_raw.get("fy", cam_raw.get("focal_length_y", 1.0)) or 1.0),
                    cx=float(cam_raw.get("cx", cam_raw.get("principal_x", 0.0)) or 0.0),
                    cy=float(cam_raw.get("cy", cam_raw.get("principal_y", 0.0)) or 0.0),
                    width=cam_raw.get("width"),
                    height=cam_raw.get("height"),
                    distortion_coefficients=cam_raw.get("distortion_coefficients"),
                    distortion_model=cam_raw.get("distortion_model"),
                )

            pose_raw = raw.get("pose") or {}
            pose = None
            if isinstance(pose_raw, dict) or pose_raw is not None:
                pose = CameraPose(
                    position=pose_raw.get("position") if isinstance(pose_raw, dict) else None,
                    orientation=pose_raw.get("orientation") if isinstance(pose_raw, dict) else None,
                    orientation_format=(
                        pose_raw.get("orientation_format", "QUATERNION_XYZW")
                        if isinstance(pose_raw, dict)
                        else "QUATERNION_XYZW"
                    ),
                )

            loc_raw = raw.get("localization") or {}
            localization = LocalizationInfo(
                status=str(loc_raw.get("status", "estimated")),
                source=list(loc_raw.get("source", ["visual"])),
                confidence=float(loc_raw.get("confidence", 1.0) or 1.0),
            )

            feats_raw = raw.get("features", []) or []
            features: List[FeatureObservation] = []
            for f in feats_raw:
                if not isinstance(f, dict):
                    continue
                features.append(
                    FeatureObservation(
                        feature_id=str(f.get("feature_id", "")),
                        xy=f.get("xy") or f.get("point") or (0.0, 0.0),
                        rgb=f.get("rgb") or f.get("color"),
                        track_id=f.get("track_id"),
                        descriptor=f.get("descriptor"),
                        response=f.get("response"),
                    )
                )

            observations.append(
                S2Observation(
                    observation_id=str(raw.get("observation_id") or raw.get("frame_id") or ""),
                    timestamp=float(raw.get("timestamp", 0.0) or 0.0),
                    image_path=str(raw.get("image") or raw.get("image_path") or ""),
                    camera=camera,
                    pose=pose,
                    features=features,
                    localization=localization,
                )
            )

        return cls(
            observations=observations,
            job_id=data.get("job_id"),
            coordinate_frame=str(data.get("coordinate_frame", "local")),
            units=str(data.get("units", "meters")),
            schema_version=str(data.get("schema_version", "1.0.0")),
            metadata=dict(data.get("metadata", {})),
        )
