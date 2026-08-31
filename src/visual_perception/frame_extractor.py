"""S1 Frame Extractor.

Extracts individual frames from UAV video input and creates structured Frame records.
Validates input video stream and extracts metadata before extraction.
"""

from typing import List, Optional

from .config import S1Config
from .logger import get_logger
from .metadata_extractor import MetadataExtractor
from .types import Frame, VideoMetadata, VideoMetadataRecord
from .video_validator import VideoValidator


class FrameExtractor:
    """Extract frames from UAV video source."""

    def __init__(self, video_path: Optional[str] = None, frame_rate: float = 2.0, config: Optional[S1Config] = None):
        """Initialize a frame extractor for a video.

        Parameters:
            video_path (Optional[str]): Path to the UAV video.
            frame_rate (float): Frame extraction rate in frames per second.
            config (Optional[S1Config]): Subsystem configuration object.
        """
        self.config = config or S1Config()
        self.video_path = video_path or self.config.video_path
        self.frame_rate = frame_rate or self.config.frame_rate
        self.logger = get_logger(self.__class__.__name__, log_level=self.config.log_level)
        self.validator = VideoValidator(log_level=self.config.log_level)
        self.metadata_extractor = MetadataExtractor(log_level=self.config.log_level)
        self.video_metadata: Optional[VideoMetadata] = None
        self.metadata_record: Optional[VideoMetadataRecord] = None

    def validate(self, video_path: Optional[str] = None) -> VideoMetadata:
        """Validate the UAV video and cache metadata.

        Parameters:
            video_path (Optional[str]): Optional override path for the video.

        Returns:
            VideoMetadata: Validated video metadata.
        """
        target_path = video_path or self.video_path
        if not target_path:
            from .exceptions import VideoNotFoundError
            raise VideoNotFoundError("No video path provided to FrameExtractor for validation.")

        self.metadata_record = self.metadata_extractor.extract(
            video_path=target_path,
            sidecar_path=self.config.telemetry_path,
            start_time_offset=self.config.time_start,
        )
        self.video_metadata = self.metadata_record.video
        self.video_path = target_path
        return self.video_metadata

    def extract(self, start_time: float = 0.0, end_time: Optional[float] = None) -> List[Frame]:
        """Extract frames from the configured video within an optional time range.

        Parameters:
            start_time (float): Beginning of the extraction range in seconds.
            end_time (Optional[float]): End of extraction range in seconds, or None for end of video.

        Returns:
            List[Frame]: List of extracted Frame objects.
        """
        if not self.video_path:
            self.logger.warning("No video path provided to FrameExtractor.")
            return []

        # Validate input video stream prior to extraction
        if self.video_metadata is None or self.video_metadata.video_path != self.video_path:
            self.validate()

        self.logger.info(
            "Extracting frames from '%s' (%dx%d @ %.2f FPS) at target %.2f FPS (start=%.2f, end=%s)",
            self.video_metadata.filename,
            self.video_metadata.width,
            self.video_metadata.height,
            self.video_metadata.fps,
            self.frame_rate,
            start_time,
            str(end_time),
        )
        return []