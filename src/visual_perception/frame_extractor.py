"""S1 Frame Extractor.

Extracts individual frames from UAV video input using configurable sampling intervals,
computes precise capture timestamps, saves images with zero-padded identifiers, and creates
structured Frame records conforming to S1 -> S2 interface specifications.
"""

from pathlib import Path
from typing import List, Optional

import cv2

from .config import S1Config
from .exceptions import VideoNotFoundError, VideoUnreadableError
from .identifier import ObservationIdentifier
from .logger import get_logger
from .metadata_extractor import MetadataExtractor
from .timestamp_handler import TimestampHandler
from .types import Frame, VideoMetadata, VideoMetadataRecord
from .video_validator import VideoValidator


class FrameExtractor:
    """Extract and sample frames from UAV video sources."""

    def __init__(
        self,
        video_path: Optional[str] = None,
        frame_rate: Optional[float] = None,
        sampling_interval: Optional[int] = None,
        sampling_mode: Optional[str] = None,
        config: Optional[S1Config] = None,
    ):
        """Initialize the frame extractor.

        Parameters:
            video_path (Optional[str]): Path to the UAV video.
            frame_rate (Optional[float]): Target extraction frame rate (used when sampling_mode='fps').
            sampling_interval (Optional[int]): Fixed sampling interval in frames (used when sampling_mode='fixed').
            sampling_mode (Optional[str]): Sampling mode ('fixed', 'fps', 'all').
            config (Optional[S1Config]): Subsystem configuration object.
        """
        self.config = config or S1Config()
        self.video_path = video_path or self.config.video_path
        self.frame_rate = frame_rate or self.config.frame_rate
        self.sampling_interval = sampling_interval or self.config.sampling_interval
        self.sampling_mode = sampling_mode or self.config.sampling_mode

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
            raise VideoNotFoundError("No video path provided to FrameExtractor for validation.")

        self.metadata_record = self.metadata_extractor.extract(
            video_path=target_path,
            sidecar_path=self.config.telemetry_path,
            start_time_offset=self.config.time_start,
        )
        self.video_metadata = self.metadata_record.video
        self.video_path = target_path
        return self.video_metadata

    def extract(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        sampling_interval: Optional[int] = None,
        sampling_mode: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> List[Frame]:
        """Sequentially extract frames according to the configured sampling interval.

        Parameters:
            start_time (Optional[float]): Beginning of the extraction range in seconds.
            end_time (Optional[float]): End of extraction range in seconds, or None for video end.
            sampling_interval (Optional[int]): Override for fixed sampling interval.
            sampling_mode (Optional[str]): Override for sampling mode ('fixed', 'fps', 'all').
            output_dir (Optional[str]): Directory to save extracted frame images.

        Returns:
            List[Frame]: Chronologically ordered list of extracted Frame objects with validated capture timestamps.
        """
        if not self.video_path:
            self.logger.warning("No video path provided to FrameExtractor.")
            return []

        # 1. Validate stream and extract metadata
        if self.video_metadata is None or self.video_metadata.video_path != self.video_path:
            self.validate()

        fps = self.video_metadata.fps
        total_stream_frames = self.video_metadata.frame_count

        # 2. Determine time range and frame index bounds
        effective_start_time = start_time if start_time is not None else self.config.time_start
        effective_end_time = end_time if end_time is not None else self.config.time_end

        start_frame_idx = max(0, int(round(effective_start_time * fps)))
        if effective_end_time is not None and effective_end_time > effective_start_time:
            end_frame_idx = min(total_stream_frames - 1, int(round(effective_end_time * fps)))
        else:
            end_frame_idx = max(0, total_stream_frames - 1)

        # 3. Determine step size based on sampling mode
        mode = sampling_mode or self.sampling_mode
        interval = sampling_interval or self.sampling_interval

        if mode == "fixed":
            step = max(1, interval)
        elif mode == "fps":
            target_fps = self.frame_rate if self.frame_rate > 0 else 2.0
            step = max(1, int(round(fps / target_fps)))
        elif mode == "all":
            step = 1
        else:
            self.logger.warning("Unknown sampling mode '%s', defaulting to fixed interval %d", mode, interval)
            step = max(1, interval)

        # 4. Prepare output destination
        target_frames_dir = Path(output_dir) if output_dir else Path(self.config.frames_dir)
        target_frames_dir.mkdir(parents=True, exist_ok=True)

        image_ext = self.config.image_format.lstrip(".").lower()
        if image_ext not in {"jpg", "jpeg", "png"}:
            image_ext = "jpg"

        encode_params = []
        if image_ext in {"jpg", "jpeg"}:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, max(1, min(100, self.config.jpeg_quality))]
        elif image_ext == "png":
            encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]

        self.logger.info(
            "Starting frame sampling from '%s' [frames %d to %d, step=%d (mode=%s)]: saving to '%s'",
            self.video_metadata.filename,
            start_frame_idx,
            end_frame_idx,
            step,
            mode,
            target_frames_dir,
        )

        # 5. Open video stream for sequential decoding
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise VideoUnreadableError(f"Failed to open video file for extraction: '{self.video_path}'")

        extracted_frames: List[Frame] = []
        current_frame_idx = 0
        frame_counter = 1

        try:
            # Fast-forward to start frame if needed
            if start_frame_idx > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
                current_frame_idx = start_frame_idx

            while cap.isOpened() and current_frame_idx <= end_frame_idx:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                # Sample if frame falls on the sampling step
                if (current_frame_idx - start_frame_idx) % step == 0:
                    # Deterministic stable identifier generation (Phase 5)
                    frame_id = ObservationIdentifier.generate_id(frame_counter)

                    # Compute precise capture timestamp (Phase 6)
                    pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                    capture_timestamp = TimestampHandler.calculate_timestamp(
                        source_frame_idx=current_frame_idx,
                        fps=fps,
                        start_offset=self.config.time_start,
                        pos_msec=pos_msec if pos_msec > 0 else None,
                    )

                    image_filename = f"{frame_id}.{image_ext}"
                    image_path = target_frames_dir / image_filename

                    # Optional resizing
                    if self.config.target_width and self.config.target_height:
                        frame_to_save = cv2.resize(
                            frame,
                            (self.config.target_width, self.config.target_height),
                            interpolation=cv2.INTER_AREA,
                        )
                    else:
                        frame_to_save = frame

                    # Save image to disk
                    cv2.imwrite(str(image_path), frame_to_save, encode_params)

                    height, width = frame_to_save.shape[:2]
                    frame_record = Frame(
                        frame_id=frame_id,
                        timestamp=capture_timestamp,
                        image_path=str(image_path.resolve()),
                        image_width=width,
                        image_height=height,
                        camera_id="primary",
                    )
                    extracted_frames.append(frame_record)
                    frame_counter += 1

                current_frame_idx += 1

        finally:
            cap.release()

        # Validate identifier uniqueness (Phase 5)
        ObservationIdentifier.validate_unique_ids(extracted_frames)

        # Validate strict timestamp monotonicity (Phase 6)
        TimestampHandler.validate_monotonicity(extracted_frames)

        self.logger.info(
            "Frame sampling complete: Extracted %d frames from '%s' with validated timestamps",
            len(extracted_frames),
            self.video_metadata.filename,
        )
        return extracted_frames