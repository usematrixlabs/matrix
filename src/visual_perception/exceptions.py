"""S1 Custom Exception Hierarchy.

Defines explicit exception classes for video input, validation,
decoding, and metadata errors in Subsystem 1 (Visual Perception).
"""


class VideoValidationError(Exception):
    """Base exception for all S1 video input and validation errors."""
    pass


class VideoNotFoundError(VideoValidationError, FileNotFoundError):
    """Raised when the specified video file path does not exist."""
    pass


class VideoFormatError(VideoValidationError, ValueError):
    """Raised when the video format or container extension is unsupported."""
    pass


class VideoCorruptError(VideoValidationError, IOError):
    """Raised when the video file is empty, has a corrupt header, or cannot be opened by the decoder."""
    pass


class VideoUnreadableError(VideoValidationError, IOError):
    """Raised when the video decoder opens the file but fails to decode/read frames."""
    pass


class VideoMetadataError(VideoValidationError, ValueError):
    """Raised when essential video metadata (FPS, frame count, resolution, duration) is missing or invalid."""
    pass

