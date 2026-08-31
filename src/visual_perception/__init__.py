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
from .keyframe_selector import KeyframeSelector
from .logger import get_logger
from .pipeline import S1Pipeline
from .types import (
    Frame,
    Keyframe,
    S1Output,
    UAVTelemetry,
    VideoMetadata,
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
    "S1Output",
    # Configuration & Logging
    "S1Config",
    "get_logger",
    # Components & Pipeline
    "FrameExtractor",
    "KeyframeSelector",
    "VideoValidator",
    "S1Pipeline",
    # Exceptions
    "VideoValidationError",
    "VideoNotFoundError",
    "VideoFormatError",
    "VideoCorruptError",
    "VideoUnreadableError",
    "VideoMetadataError",
]