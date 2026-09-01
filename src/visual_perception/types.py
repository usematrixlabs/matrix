"""S1 Type Definitions and Interface Contracts.

Data models representing visual observations, frames, keyframes,
validated video metadata, timing information, optional camera/flight/sensor
metadata, visual quality assessment, and UAV sensor telemetry,
conforming to the S1 -> S2 contract (docs/architecture/contracts/perception-localization.md).
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class QualityAssessment:
    """Represents the visual quality condition of an individual observation (Phase 7)."""

    status: str = "GOOD"  # "GOOD", "BLURRY", "OVEREXPOSED", "UNDEREXPOSED", "LOW_FEATURE", "CORRUPTED"
    blur_score: float = 0.0  # Laplacian variance (higher = sharper)
    exposure_mean: float = 0.0  # Mean grayscale brightness (0-255)
    entropy: float = 0.0  # Shannon entropy of pixel intensities
    feature_count: int = 0  # Number of detected keypoints / corners
    is_corrupted: bool = False
    quality_score: float = 100.0  # Normalized quality metric (0-100)
    flags: List[str] = field(default_factory=list)  # Multiple conditions if present

    def to_dict(self) -> Dict[str, Any]:
        """Serialize quality assessment to dictionary."""
        return asdict(self)


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
class FrameTimingInfo:
    """Represents timing information and per-frame timestamp indexing for a video stream."""

    fps: float
    frame_interval_seconds: float  # dt = 1.0 / fps
    total_frames: int
    start_timestamp: float = 0.0
    end_timestamp: float = 0.0
    duration_seconds: float = 0.0

    def get_timestamp_for_frame(self, frame_index: int) -> float:
        """Calculate the monotonic timestamp (in seconds) for a given 0-indexed frame."""
        if frame_index < 0:
            raise ValueError(f"Frame index must be non-negative, got {frame_index}")
        return round(self.start_timestamp + (frame_index * self.frame_interval_seconds), 6)

    def get_frame_index_for_timestamp(self, timestamp: float) -> int:
        """Calculate the nearest frame index for a given timestamp offset."""
        if timestamp < self.start_timestamp:
            return 0
        offset = timestamp - self.start_timestamp
        idx = int(round(offset / self.frame_interval_seconds))
        return min(idx, max(0, self.total_frames - 1))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize timing info to dictionary."""
        return asdict(self)


@dataclass
class CameraMetadata:
    """Represents optional camera and optical characteristics (explicitly None if absent)."""

    camera_id: Optional[str] = "primary"
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    focal_length_mm: Optional[float] = None
    sensor_width_mm: Optional[float] = None
    sensor_height_mm: Optional[float] = None
    field_of_view_deg: Optional[float] = None
    lens_parameters: Optional[Dict[str, Any]] = None
    exposure_mode: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize camera metadata to dictionary."""
        return asdict(self)


@dataclass
class FlightMetadata:
    """Represents optional flight mission and aircraft metadata (explicitly None if absent)."""

    flight_id: Optional[str] = None
    aircraft_model: Optional[str] = None
    takeoff_timestamp: Optional[float] = None
    pilot_operator: Optional[str] = None
    mission_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize flight metadata to dictionary."""
        return asdict(self)


@dataclass
class SensorMetadata:
    """Represents optional sensor availability and sampling rates (explicitly None/False if absent)."""

    has_gps: bool = False
    has_imu: bool = False
    has_rtk: bool = False
    gps_sampling_rate_hz: Optional[float] = None
    imu_sampling_rate_hz: Optional[float] = None
    coordinate_system: Optional[str] = "WGS84"
    altitude_reference: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize sensor metadata to dictionary."""
        return asdict(self)


@dataclass
class VideoMetadataRecord:
    """Structured internal representation of the source video and associated metadata (Phase 3 deliverable)."""

    video: VideoMetadata
    timing: FrameTimingInfo
    camera: CameraMetadata = field(default_factory=CameraMetadata)
    flight: FlightMetadata = field(default_factory=FlightMetadata)
    sensor: SensorMetadata = field(default_factory=SensorMetadata)
    source_file: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize complete structured video representation."""
        return {
            "video": {
                "width": self.video.width,
                "height": self.video.height,
                "fps": self.video.fps,
                "frame_count": self.video.frame_count,
                "duration_sec": self.video.duration_seconds,
                "codec": self.video.codec,
            },
            "timing": self.timing.to_dict(),
            "camera": self.camera.to_dict(),
            "flight": self.flight.to_dict(),
            "sensor": self.sensor.to_dict(),
            "source_file": self.source_file or self.video.video_path,
        }


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
    quality: Optional[QualityAssessment] = None
    is_keyframe: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize frame to dictionary."""
        d = asdict(self)
        if self.quality:
            d["quality"] = self.quality.to_dict()
        return d


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
