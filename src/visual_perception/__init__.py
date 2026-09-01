"""S1 — Visual Perception.

Transforms UAV video into usable visual observations and preserves
input information required by downstream subsystems.
"""

from .benchmark import S1BenchmarkRunner
from .camera_calibrator import CameraCalibrationLoader
from .config import S1Config
from .diagnostics import S1DiagnosticsEvaluator
from .downstream_validator import DownstreamValidator
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
from .packager import ObservationPackager
from .pipeline import S1Pipeline
from .quality_assessor import QualityAssessor
from .timestamp_handler import TimestampHandler
from .types import (
    CameraCalibration,
    CameraMetadata,
    FlightMetadata,
    Frame,
    FrameTimingInfo,
    Keyframe,
    QualityAssessment,
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
    "CameraCalibration",
    "FlightMetadata",
    "SensorMetadata",
    "VideoMetadataRecord",
    "QualityAssessment",
    "S1Output",
    # Configuration & Logging
    "S1Config",
    "get_logger",
    # Components & Pipeline
    "FrameExtractor",
    "KeyframeSelector",
    "VideoValidator",
    "MetadataExtractor",
    "CameraCalibrationLoader",
    "ObservationIdentifier",
    "TimestampHandler",
    "QualityAssessor",
    "ObservationPackager",
    "S1DiagnosticsEvaluator",
    "S1BenchmarkRunner",
    "DownstreamValidator",
    "S1Pipeline",
    # Exceptions
    "VideoValidationError",
    "VideoNotFoundError",
    "VideoFormatError",
    "VideoCorruptError",
    "VideoUnreadableError",
    "VideoMetadataError",
]