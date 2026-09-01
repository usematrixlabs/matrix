"""S1 Keyframe Selector.

Evaluates candidate observation frames and identifies keyframes containing
significant visual changes or viewpoints. Marks keyframe status ('is_keyframe: bool')
without removing non-keyframe candidate observations, maintaining reproducible
selection and keyframe density metrics.
"""

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .config import S1Config
from .logger import get_logger
from .types import Frame, Keyframe


class KeyframeSelector:
    """Selects and marks keyframes from candidate visual observations."""

    def __init__(self, frames: Optional[List[Frame]] = None, config: Optional[S1Config] = None):
        """Initialize the keyframe selector with optional frames and configuration.

        Parameters:
            frames (Optional[List[Frame]]): Extracted candidate frames.
            config (Optional[S1Config]): Subsystem configuration.
        """
        self.frames = frames or []
        self.config = config or S1Config()
        self.logger = get_logger(self.__class__.__name__, log_level=self.config.log_level)

    def _compute_histogram(self, image_path: str) -> Optional[np.ndarray]:
        """Load an image and compute normalized grayscale histogram."""
        p = Path(image_path)
        if not p.exists():
            return None
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None or img.size == 0:
            return None
        hist = cv2.calcHist([img], [0], None, [64], [0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist

    def select(self, frames: Optional[List[Frame]] = None) -> List[Keyframe]:
        """Evaluate frames, mark 'is_keyframe' on every frame, and return selected keyframes.

        Parameters:
            frames (Optional[List[Frame]]): Optional override of frames to select from.

        Returns:
            List[Keyframe]: Selected Keyframe objects preserving parent frame_id and timestamps.
        """
        target_frames = frames if frames is not None else self.frames
        if not target_frames:
            self.logger.info("No frames provided for keyframe selection.")
            return []

        # 1. Reset is_keyframe flag to False on all candidate frames
        for f in target_frames:
            f.is_keyframe = False

        method = self.config.keyframe_method
        change_thresh = self.config.keyframe_change_threshold
        min_interval = self.config.min_keyframe_interval_frames
        max_interval = self.config.max_keyframe_interval_frames
        skip_poor = self.config.skip_poor_quality_keyframes

        self.logger.info(
            "Executing keyframe detection on %d frames using method '%s' (change_threshold=%.2f, min_interval=%d, max_interval=%d)",
            len(target_frames),
            method,
            change_thresh,
            min_interval,
            max_interval,
        )

        selected_records: List[Keyframe] = []

        if method == "content_change":
            last_kf_idx = -1
            last_hist: Optional[np.ndarray] = None

            for idx, frame in enumerate(target_frames):
                is_corrupted = frame.quality and frame.quality.is_corrupted
                if skip_poor and is_corrupted:
                    continue

                # First valid frame is always a keyframe
                if last_kf_idx == -1:
                    frame.is_keyframe = True
                    hist = self._compute_histogram(frame.image_path)
                    last_hist = hist
                    last_kf_idx = idx
                    score = frame.quality.quality_score if frame.quality else 100.0
                    feat_cnt = frame.quality.feature_count if frame.quality else None
                    selected_records.append(
                        Keyframe(
                            frame_id=frame.frame_id,
                            timestamp=frame.timestamp,
                            image_path=frame.image_path,
                            score=score,
                            selection_reason="initial_reference_frame",
                            visual_features_count=feat_cnt,
                        )
                    )
                    continue

                frame_gap = idx - last_kf_idx
                hist = self._compute_histogram(frame.image_path)

                # Compute histogram visual divergence if both histograms available
                hist_dist = 0.0
                if last_hist is not None and hist is not None:
                    # Bhattacharyya distance in range [0, 1]
                    hist_dist = float(cv2.compareHist(last_hist, hist, cv2.HISTCMP_BHATTACHARYYA))

                should_select = False
                reason = "content_change"

                # Condition A: Visual change exceeds threshold and min_interval satisfied
                if frame_gap >= min_interval and hist_dist >= change_thresh:
                    should_select = True
                    reason = f"visual_content_change (dist={hist_dist:.3f})"
                # Condition B: Max interval reached (temporal timeout guarantee)
                elif frame_gap >= max_interval:
                    should_select = True
                    reason = f"max_interval_timeout (gap={frame_gap})"

                if should_select:
                    frame.is_keyframe = True
                    last_hist = hist
                    last_kf_idx = idx
                    score = frame.quality.quality_score if frame.quality else 100.0
                    feat_cnt = frame.quality.feature_count if frame.quality else None
                    selected_records.append(
                        Keyframe(
                            frame_id=frame.frame_id,
                            timestamp=frame.timestamp,
                            image_path=frame.image_path,
                            score=score,
                            selection_reason=reason,
                            visual_features_count=feat_cnt,
                        )
                    )

        elif method == "uniform":
            step = max(1, min_interval)
            for idx in range(0, len(target_frames), step):
                frame = target_frames[idx]
                if skip_poor and frame.quality and frame.quality.is_corrupted:
                    continue
                frame.is_keyframe = True
                score = frame.quality.quality_score if frame.quality else 100.0
                feat_cnt = frame.quality.feature_count if frame.quality else None
                selected_records.append(
                    Keyframe(
                        frame_id=frame.frame_id,
                        timestamp=frame.timestamp,
                        image_path=frame.image_path,
                        score=score,
                        selection_reason=f"uniform_sampling (step={step})",
                        visual_features_count=feat_cnt,
                    )
                )

        elif method == "quality_maxima":
            # Slide window and pick local max quality
            window_size = max(2, min_interval * 2)
            for start in range(0, len(target_frames), window_size):
                window = target_frames[start : start + window_size]
                valid_window = [f for f in window if not (skip_poor and f.quality and f.quality.is_corrupted)]
                if not valid_window:
                    continue
                best_frame = max(valid_window, key=lambda f: f.quality.quality_score if f.quality else 50.0)
                best_frame.is_keyframe = True
                score = best_frame.quality.quality_score if best_frame.quality else 100.0
                feat_cnt = best_frame.quality.feature_count if best_frame.quality else None
                selected_records.append(
                    Keyframe(
                        frame_id=best_frame.frame_id,
                        timestamp=best_frame.timestamp,
                        image_path=best_frame.image_path,
                        score=score,
                        selection_reason="local_quality_maximum",
                        visual_features_count=feat_cnt,
                    )
                )

        # Enforce max_keyframes if specified
        if self.config.max_keyframes and len(selected_records) > self.config.max_keyframes:
            allowed_ids = {k.frame_id for k in selected_records[: self.config.max_keyframes]}
            selected_records = selected_records[: self.config.max_keyframes]
            for f in target_frames:
                f.is_keyframe = f.frame_id in allowed_ids

        density = self.calculate_keyframe_density(target_frames, selected_records)

        self.logger.info(
            "Keyframe detection complete: Selected %d keyframes from %d candidate frames (density=%.2f%%)",
            len(selected_records),
            len(target_frames),
            density * 100.0,
        )
        return selected_records

    @staticmethod
    def calculate_keyframe_density(frames: List[Frame], keyframes: List[Keyframe]) -> float:
        """Calculate keyframe density as a fraction of total candidate observations.

        Parameters:
            frames (List[Frame]): Total candidate frames.
            keyframes (List[Keyframe]): Selected keyframes.

        Returns:
            float: Ratio of keyframes to candidate frames (0.0 to 1.0).
        """
        if not frames:
            return 0.0
        return round(len(keyframes) / len(frames), 4)