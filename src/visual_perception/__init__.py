"""S1 — Visual Perception.

Transforms UAV video into usable visual observations and preserves
input information required by downstream subsystems.
"""

from .config import S1Config
from .exceptions import (
    VideoCorruptError,
    VideoFormatError,
    VideoMetadataError,
    VideoNotFoundError,
    VideoUnreadableError,
    VideoValidationError,
)
from .frame_extractor import FrameExtractor
from .identifier import ObservationIdentifier
from .keyframe_selector import KeyframeSelector
from .logger import get_logger
from .metadata_extractor import MetadataExtractor
from .pipeline import S1Pipeline
from .timestamp_handler import TimestampHandler
from .types import (
    CameraMetadata,
    FlightMetadata,
    Frame,
    FrameTimingInfo,
    Keyframe,
    S1Output,
    SensorMetadata,
    UAVTelemetry,
    VideoMetadata,
    VideoMetadataRecord,
    VisualObservations,
)
from .video_validator import VideoValidator

__all__ = [
    # Types & Models
    "Frame",
    "Keyframe",
    "UAVTelemetry",
    "VisualObservations",
    "VideoMetadata",
    "FrameTimingInfo",
    "CameraMetadata",
    "FlightMetadata",
    "SensorMetadata",
    "VideoMetadataRecord",
    "S1Output",
    # Configuration & Logging
    "S1Config",
    "get_logger",
    # Components & Pipeline
    "FrameExtractor",
    "KeyframeSelector",
    "VideoValidator",
    "MetadataExtractor",
    "ObservationIdentifier",
    "TimestampHandler",
    "S1Pipeline",
    # Exceptions
    "VideoValidationError",
    "VideoNotFoundError",
    "VideoFormatError",
    "VideoCorruptError",
    "VideoUnreadableError",
    "VideoMetadataError",
]