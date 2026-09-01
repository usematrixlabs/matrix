"""S1 Diagnostics & Health Evaluator.

Evaluates pipeline execution status, distinguishes hard input/processing failures from
graceful degradation (e.g. sparse observations, high blur ratios, missing optional calibration/telemetry),
and compiles structured diagnostic health records conforming to Phase 11 requirements.
"""

from typing import Any, Dict, List, Optional, Tuple

from .config import S1Config
from .logger import get_logger
from .types import Frame, Keyframe, VideoMetadataRecord


class S1DiagnosticsEvaluator:
    """Evaluates S1 visual perception execution health and degradation conditions."""

    def __init__(self, config: Optional[S1Config] = None):
        """Initialize the diagnostics evaluator.

        Parameters:
            config (Optional[S1Config]): Subsystem configuration.
        """
        self.config = config or S1Config()
        self.logger = get_logger(self.__class__.__name__, log_level=self.config.log_level)

    def evaluate_health(
        self,
        frames: List[Frame],
        keyframes: List[Keyframe],
        video_record: Optional[VideoMetadataRecord] = None,
        telemetry_loaded: bool = False,
    ) -> Tuple[str, List[str], Dict[str, Any]]:
        """Evaluate visual perception health, determine status, warnings, and compile diagnostics.

        Parameters:
            frames (List[Frame]): Extracted candidate observation frames.
            keyframes (List[Keyframe]): Selected keyframes.
            video_record (Optional[VideoMetadataRecord]): Video stream and calibration metadata.
            telemetry_loaded (bool): True if external telemetry file was parsed.

        Returns:
            Tuple[str, List[str], Dict[str, Any]]: (status, warnings_list, diagnostics_dict)
        """
        warnings: List[str] = []
        errors: List[str] = []
        degraded_reasons: List[str] = []

        total_frames = len(frames)
        total_keyframes = len(keyframes)

        # 1. Quality distribution analysis
        corrupted_count = sum(1 for f in frames if f.quality and f.quality.status == "CORRUPTED")
        blurry_count = sum(1 for f in frames if f.quality and f.quality.status == "BLURRY")
        overexposed_count = sum(1 for f in frames if f.quality and f.quality.status == "OVEREXPOSED")
        underexposed_count = sum(1 for f in frames if f.quality and f.quality.status == "UNDEREXPOSED")
        low_feature_count = sum(1 for f in frames if f.quality and f.quality.status == "LOW_FEATURE")
        good_count = sum(1 for f in frames if f.quality and f.quality.status == "GOOD")

        valid_frames_count = total_frames - corrupted_count

        # 2. Check missing optional metadata (Quality Warnings)
        is_calibrated = False
        if video_record and video_record.calibration:
            is_calibrated = video_record.calibration.is_calibrated

        if not is_calibrated:
            warnings.append("missing_camera_calibration: operating with uncalibrated pinhole assumptions")

        if not telemetry_loaded:
            warnings.append("missing_uav_telemetry: operating in visual-only mode")

        # 3. Evaluate failure vs degradation vs completed status
        min_obs = getattr(self.config, "min_valid_observations", 1)
        max_deg_ratio = getattr(self.config, "max_degraded_ratio", 1.0)

        if total_frames > 0 and valid_frames_count == 0:
            status = "failed"
            errors.append("all_observations_corrupted: no usable visual frames extracted")
            self.logger.error("S1 Failure: All %d extracted observations are corrupted.", total_frames)
        elif 0 < valid_frames_count < min_obs:
            status = "degraded"
            reason = f"insufficient_valid_observations: {valid_frames_count} frames available (minimum threshold: {min_obs})"
            warnings.append(reason)
            degraded_reasons.append(reason)
            self.logger.warning("S1 Degraded: %s", reason)
        elif max_deg_ratio < 1.0:
            # Check degradation ratio among observations when threshold is configured (< 1.0)
            degraded_count = blurry_count + corrupted_count + low_feature_count
            degraded_ratio = (degraded_count / total_frames) if total_frames > 0 else 0.0

            if total_frames > 0 and degraded_ratio >= max_deg_ratio:
                status = "degraded"
                reason = f"high_visual_degradation_ratio: {degraded_ratio * 100.0:.1f}% of observations are blurry or low-feature (threshold: {max_deg_ratio * 100.0:.1f}%)"
                warnings.append(reason)
                degraded_reasons.append(reason)
                self.logger.warning("S1 Degraded: %s", reason)
            else:
                status = "completed"
        elif corrupted_count > 0:
            status = "degraded"
            reason = f"corrupted_observations_detected: {corrupted_count} frames were corrupted"
            warnings.append(reason)
            degraded_reasons.append(reason)
        else:
            status = "completed"

        # 4. Compile health diagnostics
        diagnostics = {
            "health_status": status,
            "is_valid": status != "failed",
            "is_degraded": status == "degraded",
            "degraded_reasons": degraded_reasons,
            "observations_summary": {
                "total_extracted": total_frames,
                "valid_count": valid_frames_count,
                "corrupted_count": corrupted_count,
                "good_count": good_count,
                "blurry_count": blurry_count,
                "overexposed_count": overexposed_count,
                "underexposed_count": underexposed_count,
                "low_feature_count": low_feature_count,
                "keyframes_selected": total_keyframes,
                "keyframe_density": round(total_keyframes / total_frames, 4) if total_frames > 0 else 0.0,
            },
            "sensor_availability": {
                "camera_calibration": is_calibrated,
                "telemetry_present": telemetry_loaded,
            },
        }

        return status, warnings, diagnostics

