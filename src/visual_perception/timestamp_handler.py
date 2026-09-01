"""S1 Timestamp Handler.

Calculates, preserves, and validates monotonic capture timestamps for video observations,
ensuring strict chronological alignment for downstream S2 localization and sensor fusion.
"""

from typing import Any, Dict, List, Optional

from .types import Frame, Keyframe


class TimestampHandler:
    """Utility for calculating and validating observation capture timestamps."""

    TIME_UNIT: str = "seconds"
    DEFAULT_PRECISION: int = 6  # Microsecond precision for float seconds

    @classmethod
    def calculate_timestamp(
        cls,
        source_frame_idx: int,
        fps: float,
        start_offset: float = 0.0,
        base_time: Optional[float] = None,
        pos_msec: Optional[float] = None,
    ) -> float:
        """Calculate capture timestamp in seconds for a specific source frame.

        Parameters:
            source_frame_idx (int): 0-based frame index in the source video stream.
            fps (float): Video stream frame rate (FPS).
            start_offset (float): Configured start time offset in seconds.
            base_time (Optional[float]): Optional absolute UTC epoch timestamp at takeoff / start.
            pos_msec (Optional[float]): Optional container presentation timestamp (PTS) in milliseconds.

        Returns:
            float: Monotonic capture timestamp in seconds.

        Raises:
            ValueError: If source_frame_idx < 0 or fps <= 0.
        """
        if source_frame_idx < 0:
            raise ValueError(f"Source frame index must be non-negative, got {source_frame_idx}")
        if fps <= 0.0:
            raise ValueError(f"FPS must be strictly positive, got {fps}")

        # If presentation timestamp (PTS) is valid and non-negative, prioritize stream timing
        if pos_msec is not None and pos_msec >= 0.0:
            stream_time = pos_msec / 1000.0
        else:
            stream_time = source_frame_idx / fps

        # Combine with start offset and optional absolute base time
        base = base_time if base_time is not None else 0.0
        total_time = base + start_offset + stream_time
        return round(total_time, cls.DEFAULT_PRECISION)

    @classmethod
    def validate_monotonicity(cls, frames: List[Frame]) -> bool:
        """Validate that timestamps across observations are strictly monotonically increasing.

        Parameters:
            frames (List[Frame]): Chronological list of frames to check.

        Returns:
            bool: True if strictly monotonic.

        Raises:
            ValueError: If any timestamp is out of order, negative, or duplicate.
        """
        if not frames:
            return True

        prev_time: Optional[float] = None
        for idx, frame in enumerate(frames):
            t = frame.timestamp
            if t is None or not isinstance(t, (int, float)):
                raise ValueError(f"Observation at index {idx} ({frame.frame_id}) has invalid timestamp: {t!r}")

            if t < 0.0:
                raise ValueError(f"Observation at index {idx} ({frame.frame_id}) has negative timestamp: {t}")

            if prev_time is not None:
                if t <= prev_time:
                    raise ValueError(
                        f"Non-monotonic timestamp detected at index {idx} ({frame.frame_id}): "
                        f"current timestamp {t:.6f}s is not strictly greater than previous {prev_time:.6f}s"
                    )

            prev_time = t

        return True

    @classmethod
    def to_observation_timing_dict(cls, frame: Frame) -> Dict[str, Any]:
        """Generate a standardized timing record for an observation.

        Parameters:
            frame (Frame): Observation frame record.

        Returns:
            Dict[str, Any]: Standardized observation timing dictionary.
        """
        return {
            "observation_id": frame.frame_id,
            "timestamp": frame.timestamp,
            "unit": cls.TIME_UNIT,
        }

