"""S1 Stable Observation Identifier.

Generates and validates deterministic, unique, and stable observation identifiers
(e.g., 'frame_000001') that remain associated with frames and keyframes throughout
the Matrix pipeline (S1 -> S2 -> S3).
"""

import re
from typing import List, Optional, Set

from .types import Frame, Keyframe


class ObservationIdentifier:
    """Utility for generating and validating stable observation identifiers."""

    DEFAULT_PREFIX: str = "frame_"
    DEFAULT_WIDTH: int = 6

    @classmethod
    def generate_id(cls, index: int, prefix: Optional[str] = None, width: Optional[int] = None) -> str:
        """Generate a deterministic, zero-padded observation identifier.

        Parameters:
            index (int): 1-based sequence index of the observation (e.g. 1, 2, ...).
            prefix (Optional[str]): String prefix (defaults to 'frame_').
            width (Optional[int]): Zero-padding width (defaults to 6).

        Returns:
            str: Deterministic identifier string (e.g. 'frame_000001').

        Raises:
            ValueError: If index is less than 1.
        """
        if index < 1:
            raise ValueError(f"Observation index must be a positive integer (>= 1), got {index}")

        p = prefix if prefix is not None else cls.DEFAULT_PREFIX
        w = width if width is not None else cls.DEFAULT_WIDTH
        return f"{p}{index:0{w}d}"

    @classmethod
    def parse_id(cls, frame_id: str, prefix: Optional[str] = None) -> int:
        """Extract the numeric sequence index from an observation identifier.

        Parameters:
            frame_id (str): Formatted observation identifier (e.g. 'frame_000042').
            prefix (Optional[str]): Expected string prefix.

        Returns:
            int: Integer sequence index (e.g. 42).

        Raises:
            ValueError: If the identifier does not match the expected pattern.
        """
        p = prefix if prefix is not None else cls.DEFAULT_PREFIX
        pattern = rf"^{re.escape(p)}(\d+)$"
        match = re.match(pattern, frame_id)
        if not match:
            raise ValueError(f"Identifier '{frame_id}' does not match expected format '{p}<digits>'")
        return int(match.group(1))

    @classmethod
    def is_valid_format(cls, frame_id: str, prefix: Optional[str] = None, width: Optional[int] = None) -> bool:
        """Check if an identifier matches the standard format.

        Parameters:
            frame_id (str): Identifier to check.
            prefix (Optional[str]): Expected prefix (defaults to 'frame_').
            width (Optional[int]): Expected padding width (defaults to 6).

        Returns:
            bool: True if valid, False otherwise.
        """
        p = prefix if prefix is not None else cls.DEFAULT_PREFIX
        w = width if width is not None else cls.DEFAULT_WIDTH
        pattern = rf"^{re.escape(p)}\d{{{w}}}$"
        return bool(re.match(pattern, frame_id))

    @classmethod
    def validate_unique_ids(cls, frames: List[Frame]) -> bool:
        """Validate that all frames in the observation set have unique, non-empty identifiers.

        Parameters:
            frames (List[Frame]): List of frames to validate.

        Returns:
            bool: True if all IDs are unique and valid.

        Raises:
            ValueError: If empty or duplicate identifiers are detected.
        """
        seen_ids: Set[str] = set()
        for idx, frame in enumerate(frames):
            if not frame.frame_id or not isinstance(frame.frame_id, str):
                raise ValueError(f"Frame at index {idx} has an invalid or empty frame_id: {frame.frame_id!r}")

            if frame.frame_id in seen_ids:
                raise ValueError(f"Duplicate frame_id detected: '{frame.frame_id}' at observation index {idx}")

            seen_ids.add(frame.frame_id)

        return True

