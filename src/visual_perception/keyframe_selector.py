"""S1 Keyframe Selector.

Evaluates extracted frames and selects keyframes based on quality and criteria.
"""

from typing import List, Optional

from .config import S1Config
from .logger import get_logger
from .types import Frame, Keyframe


class KeyframeSelector:
    """Select keyframes from extracted frames."""

    def __init__(self, frames: Optional[List[Frame]] = None, config: Optional[S1Config] = None):
        """Initialize a keyframe selector with extracted frames and optional configuration.

        Parameters:
            frames (Optional[List[Frame]]): Extracted frames to evaluate.
            config (Optional[S1Config]): Subsystem configuration.
        """
        self.frames = frames or []
        self.config = config or S1Config()
        self.logger = get_logger(self.__class__.__name__, log_level=self.config.log_level)

    def select(self, frames: Optional[List[Frame]] = None) -> List[Keyframe]:
        """Select keyframes from candidate frames.

        Parameters:
            frames (Optional[List[Frame]]): Optional override of frames to select from.

        Returns:
            List[Keyframe]: List of selected Keyframe objects.
        """
        target_frames = frames if frames is not None else self.frames
        if not target_frames:
            self.logger.info("No frames provided for keyframe selection.")
            return []

        self.logger.info(
            "Selecting keyframes from %d candidate frames using method '%s' (threshold=%.2f)",
            len(target_frames),
            self.config.keyframe_method,
            self.config.quality_threshold,
        )
        # Initial stub implementation: keyframe scoring/selection algorithms implemented in Phase 2
        return []