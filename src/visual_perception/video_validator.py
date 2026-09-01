"""S1 Video Validator.

Performs robust input verification on UAV video files, extracts essential stream
metadata, and raises explicit errors on missing, corrupt, unsupported, or unreadable inputs.
"""

import os
from pathlib import Path
from typing import Optional, Set

import cv2

from .exceptions import (
    VideoCorruptError,
    VideoFormatError,
    VideoMetadataError,
    VideoNotFoundError,
    VideoUnreadableError,
)
from .logger import get_logger
from .types import VideoMetadata


class VideoValidator:
    """Validates UAV input videos and extracts stream metadata."""

    SUPPORTED_EXTENSIONS: Set[str] = {
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".m4v",
        ".webm",
        ".ts",
    }

    def __init__(self, log_level: str = "INFO"):
        """Initialize the video validator."""
        self.logger = get_logger(self.__class__.__name__, log_level=log_level)

    def validate(self, video_path: str) -> VideoMetadata:
        """Validate the UAV video at the specified path and extract metadata.

        Parameters:
            video_path (str): File path to the video.

        Returns:
            VideoMetadata: Validated video metadata object.

        Raises:
            VideoNotFoundError: If the file does not exist.
            VideoFormatError: If the file extension/container is unsupported.
            VideoCorruptError: If the file is 0 bytes or the video decoder cannot open it.
            VideoUnreadableError: If the decoder opens the file but cannot read frames.
            VideoMetadataError: If essential video properties (FPS, duration, resolution) are missing/invalid.
        """
        if not video_path:
            raise VideoNotFoundError("No video path provided for validation.")

        video_file = Path(video_path)

        # 1. Check file existence
        if not video_file.is_file():
            raise VideoNotFoundError(
                f"Video file not found at path: '{video_path}'"
            )

        # 2. Check file size
        file_size = video_file.stat().st_size
        if file_size == 0:
            raise VideoCorruptError(
                f"Video file is empty (0 bytes): '{video_path}'"
            )

        # 3. Check container extension
        extension = video_file.suffix.lower()
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise VideoFormatError(
                f"Unsupported video container format '{extension}' for file '{video_file.name}'. "
                f"Supported formats: {', '.join(sorted(self.SUPPORTED_EXTENSIONS))}"
            )

        # 4. Decoder open check via OpenCV
        cap = cv2.VideoCapture(str(video_file.resolve()))
        if not cap.isOpened():
            cap.release()
            raise VideoCorruptError(
                f"Failed to open video stream or codec header is corrupt for: '{video_path}'"
            )

        try:
            # 5. Extract stream properties
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))

            # Decode FourCC codec identifier
            codec = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]).strip()
            if not codec or any(ord(c) < 32 or ord(c) > 126 for c in codec):
                codec = "unknown"

            # 6. Readability check: verify at least the first frame decodes properly
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                raise VideoUnreadableError(
                    f"Video decoder opened '{video_file.name}', but failed to read or decode frames."
                )

            # Check for valid dimensions
            if width <= 0 or height <= 0:
                # Attempt to retrieve from decoded frame
                height, width = frame.shape[:2]

            if width <= 0 or height <= 0:
                raise VideoMetadataError(
                    f"Invalid video resolution: {width}x{height} for '{video_file.name}'"
                )

            # Check FPS validity
            if fps <= 0.0 or fps > 1000.0:
                raise VideoMetadataError(
                    f"Invalid or missing frame rate (FPS={fps}) in video '{video_file.name}'"
                )

            # Check frame count / calculate duration
            if frame_count <= 0:
                # In some stream formats, frame count is not in header; at least 1 frame was decoded
                frame_count = 1

            duration_seconds = frame_count / fps
            if duration_seconds <= 0:
                raise VideoMetadataError(
                    f"Invalid video duration calculated: {duration_seconds:.2f}s for '{video_file.name}'"
                )

            notes = [
                f"Codec: {codec}",
                f"Resolution: {width}x{height}",
                f"FPS: {fps:.2f}",
                f"Total Frames: {frame_count}",
                f"Duration: {duration_seconds:.2f}s",
            ]

            metadata = VideoMetadata(
                video_path=str(video_file.resolve()),
                filename=video_file.name,
                file_size_bytes=file_size,
                frame_count=frame_count,
                fps=fps,
                width=width,
                height=height,
                duration_seconds=round(duration_seconds, 3),
                codec=codec,
                is_valid=True,
                validation_notes=notes,
            )

            self.logger.info(
                "Successfully validated video '%s': %dx%d @ %.2f FPS, %d frames (%.2fs)",
                video_file.name,
                width,
                height,
                fps,
                frame_count,
                duration_seconds,
            )
            return metadata

        finally:
            cap.release()

