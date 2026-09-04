"""
S3 Failure Modes Taxonomy and Exceptions
"""

from enum import Enum
from typing import Optional


class S3FailureReason(str, Enum):
    """Categorized failure reasons for S3 reconstruction."""
    INSUFFICIENT_OBSERVATIONS = "INSUFFICIENT_OBSERVATIONS"
    DEGENERATE_CAMERA_GEOMETRY = "DEGENERATE_CAMERA_GEOMETRY"
    NO_TRIANGULABLE_TRACKS = "NO_TRIANGULABLE_TRACKS"
    HIGH_REPROJECTION_ERROR = "HIGH_REPROJECTION_ERROR"
    CHEIRALITY_VIOLATION = "CHEIRALITY_VIOLATION"
    CORRUPT_INPUT = "CORRUPT_INPUT"
    INTERNAL_COMPUTATION_ERROR = "INTERNAL_COMPUTATION_ERROR"


class S3ReconstructionError(Exception):
    """Raised when S3 reconstruction fails unrecoverably."""

    def __init__(self, reason: S3FailureReason, message: str, details: Optional[dict] = None) -> None:
        super().__init__(f"[{reason.value}] {message}")
        self.reason = reason
        self.details = details or {}

