"""S1 Keyframe Selector.

Evaluates extracted candidate frames and selects keyframes for downstream processing.
Ensures every selected Keyframe preserves the exact, stable frame_id of its source observation.
"""

from typing import List, Optional

from .config import S1Config
from .logger import get_logger
from .types import Frame, Keyframe


class KeyframeSelector:
    """Select keyframes from extracted candidate frames."""

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
        """Select keyframes from candidate frames, preserving observation IDs.

        Parameters:
            frames (Optional[List[Frame]]): Optional override of frames to select from.

        Returns:
            List[Keyframe]: List of selected Keyframe objects with stable frame_id association.
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

        selected_keyframes: List[Keyframe] = []
        method = self.config.keyframe_method

        if method == "uniform":
            # Subsample candidate frames uniformly as keyframes (e.g. every 2nd or all if few)
            step = 2 if len(target_frames) > 10 else 1
            for idx in range(0, len(target_frames), step):
                f = target_frames[idx]
                keyframe = Keyframe(
                    frame_id=f.frame_id,  # Preserve the exact stable observation ID
                    timestamp=f.timestamp,
                    image_path=f.image_path,
                    score=100.0,
                    selection_reason="uniform_candidate_selection",
                )
                selected_keyframes.append(keyframe)
        else:
            # Default fallback: promote candidate frames with original frame_ids
            for f in target_frames:
                keyframe = Keyframe(
                    frame_id=f.frame_id,
                    timestamp=f.timestamp,
                    image_path=f.image_path,
                    score=100.0,
                    selection_reason="direct_promotion",
                )
                selected_keyframes.append(keyframe)

        if self.config.max_keyframes and len(selected_keyframes) > self.config.max_keyframes:
            selected_keyframes = selected_keyframes[:self.config.max_keyframes]

        self.logger.info(
            "Selected %d keyframes from %d candidate frames (IDs preserved)",
            len(selected_keyframes),
            len(target_frames),
        )
        return selected_keyframes