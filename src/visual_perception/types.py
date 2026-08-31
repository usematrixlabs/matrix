"""S1 Type Definitions and Interface Contracts.

Data models representing visual observations, frames, keyframes,
validated video metadata, and optional UAV sensor telemetry,
conforming to the S1 -> S2 contract (docs/architecture/contracts/perception-localization.md).
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class VideoMetadata:
    """Represents validated video stream information extracted during Phase 2."""

    video_path: str
    filename: str
    file_size_bytes: int
    frame_count: int
    fps: float
    width: int
    height: int
    duration_seconds: float
    codec: str
    is_valid: bool = True
    validation_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize video metadata to dictionary."""
        return asdict(self)


@dataclass
class Frame:
    """Represents an individual video frame extracted during S1."""

    frame_id: str
    timestamp: float
    image_path: str
    image_width: int
    image_height: int
    exposure_time: Optional[float] = None
    camera_id: Optional[str] = "primary"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize frame to dictionary."""
        return asdict(self)


@dataclass
class Keyframe:
    """Represents a selected keyframe for downstream feature matching / reconstruction."""

    frame_id: str
    timestamp: float
    image_path: str
    score: Optional[float] = None
    selection_reason: Optional[str] = None
    visual_features_count: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize keyframe to dictionary."""
        return asdict(self)


@dataclass
class UAVTelemetry:
    """Represents optional UAV sensor telemetry passed through without S1 interpretation."""

    gps_coordinates: Optional[Dict[str, float]] = None
    gnss_status: Optional[str] = None
    imu_data: Optional[Dict[str, Any]] = None
    altitude: Optional[float] = None
    rtk_ppk: Optional[Dict[str, Any]] = None
    flight_telemetry: Optional[Dict[str, Any]] = None
    sensor_measurements: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize telemetry data to dictionary."""
        return asdict(self)


@dataclass
class VisualObservations:
    """Container for all visual observations extracted from UAV video."""

    frames: List[Frame] = field(default_factory=list)
    keyframes: List[Keyframe] = field(default_factory=list)
    frame_ordering: List[str] = field(default_factory=list)
    visual_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize visual observations to dictionary."""
        return {
            "frames": [f.to_dict() for f in self.frames],
            "keyframes": [k.to_dict() for k in self.keyframes],
            "frame_ordering": self.frame_ordering,
            "visual_metadata": self.visual_metadata,
        }


@dataclass
class S1Output:
    """Top-level S1 output contract conforming to S1 -> S2 interface."""

    visual_observations: VisualObservations = field(default_factory=VisualObservations)
    temporal_information: Dict[str, Any] = field(default_factory=dict)
    available_uav_information: UAVTelemetry = field(default_factory=UAVTelemetry)
    status: str = "initialized"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize S1 output contract to dictionary."""
        return {
            "visual_observations": self.visual_observations.to_dict(),
            "temporal_information": self.temporal_information,
            "available_uav_information": self.available_uav_information.to_dict(),
            "status": self.status,
            "metadata": self.metadata,
        }
