"""S1 Visual Quality Assessor.

Evaluates candidate observation frames for blur, exposure extremes,
low-feature content, and corruption. Generates structured QualityAssessment
records without discarding frames prematurely.
"""

from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .config import S1Config
from .logger import get_logger
from .types import QualityAssessment


class QualityAssessor:
    """Evaluates visual quality of candidate observation frames."""

    def __init__(
        self,
        blur_threshold: float = 100.0,
        underexposure_threshold: float = 30.0,
        overexposure_threshold: float = 230.0,
        low_feature_threshold: int = 50,
        min_entropy_threshold: float = 3.5,
        config: Optional[S1Config] = None,
    ):
        """Initialize the visual quality assessor.

        Parameters:
            blur_threshold (float): Minimum Laplacian variance for sharpness.
            underexposure_threshold (float): Minimum mean grayscale brightness (0-255).
            overexposure_threshold (float): Maximum mean grayscale brightness (0-255).
            low_feature_threshold (int): Minimum number of detected FAST keypoints.
            min_entropy_threshold (float): Minimum Shannon entropy for texture richness.
            config (Optional[S1Config]): Subsystem configuration object.
        """
        self.config = config or S1Config()
        self.blur_threshold = getattr(self.config, "blur_threshold", blur_threshold)
        self.underexposure_threshold = getattr(self.config, "underexposure_threshold", underexposure_threshold)
        self.overexposure_threshold = getattr(self.config, "overexposure_threshold", overexposure_threshold)
        self.low_feature_threshold = getattr(self.config, "low_feature_threshold", low_feature_threshold)
        self.min_entropy_threshold = getattr(self.config, "min_entropy_threshold", min_entropy_threshold)
        self.logger = get_logger(self.__class__.__name__, log_level=self.config.log_level)

    def assess_frame(self, image: Optional[np.ndarray], frame_id: Optional[str] = None) -> QualityAssessment:
        """Evaluate the visual quality of an image array.

        Parameters:
            image (Optional[np.ndarray]): Input BGR or Grayscale pixel array.
            frame_id (Optional[str]): Observation identifier for logging.

        Returns:
            QualityAssessment: Structured assessment results with metrics and status.
        """
        # 1. Check for corruption
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return QualityAssessment(
                status="CORRUPTED",
                is_corrupted=True,
                quality_score=0.0,
                flags=["CORRUPTED"],
            )

        if np.isnan(image).any() or np.isinf(image).any():
            return QualityAssessment(
                status="CORRUPTED",
                is_corrupted=True,
                quality_score=0.0,
                flags=["CORRUPTED"],
            )

        # 2. Convert to Grayscale
        if len(image.shape) == 3 and image.shape[2] >= 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 2:
            gray = image
        else:
            return QualityAssessment(
                status="CORRUPTED",
                is_corrupted=True,
                quality_score=0.0,
                flags=["CORRUPTED"],
            )

        # 3. Compute Blur Score (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = float(laplacian.var())

        # 4. Compute Exposure (Mean intensity)
        exposure_mean = float(np.mean(gray))

        # 5. Compute Shannon Entropy (Texture richness)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
        hist_sum = hist.sum()
        if hist_sum > 0:
            p = hist / hist_sum
            non_zero = p[p > 0]
            entropy = float(-np.sum(non_zero * np.log2(non_zero)))
        else:
            entropy = 0.0

        # 6. Compute Feature Count (FAST Corners or Harris)
        feature_count = 0
        try:
            detector = cv2.FastFeatureDetector_create(threshold=15)
            keypoints = detector.detect(gray, None)
            feature_count = len(keypoints)
        except Exception:
            corners = cv2.goodFeaturesToTrack(gray, maxCorners=500, qualityLevel=0.01, minDistance=5)
            feature_count = len(corners) if corners is not None else 0

        # 7. Evaluate against thresholds
        flags: List[str] = []
        is_extreme_exposure = False

        if exposure_mean > self.overexposure_threshold:
            flags.append("OVEREXPOSED")
            is_extreme_exposure = True
        elif exposure_mean < self.underexposure_threshold:
            flags.append("UNDEREXPOSED")
            is_extreme_exposure = True

        if not is_extreme_exposure:
            if blur_score < self.blur_threshold:
                flags.append("BLURRY")
            if feature_count < self.low_feature_threshold:
                flags.append("LOW_FEATURE")
        else:
            if feature_count < self.low_feature_threshold:
                flags.append("LOW_FEATURE")

        # Determine primary status
        if not flags:
            primary_status = "GOOD"
        elif "OVEREXPOSED" in flags:
            primary_status = "OVEREXPOSED"
        elif "UNDEREXPOSED" in flags:
            primary_status = "UNDEREXPOSED"
        elif "BLURRY" in flags:
            # If completely flat (zero features and low entropy), categorize as LOW_FEATURE
            if feature_count < 10 and entropy < 2.0:
                primary_status = "LOW_FEATURE"
            else:
                primary_status = "BLURRY"
        elif "LOW_FEATURE" in flags:
            primary_status = "LOW_FEATURE"
        else:
            primary_status = flags[0]

        # 8. Compute normalized 0-100 quality score
        sharp_comp = min(40.0, (blur_score / max(1.0, self.blur_threshold)) * 40.0)
        exp_diff = abs(exposure_mean - 128.0)
        exp_comp = max(0.0, 30.0 - (exp_diff / 128.0) * 30.0)
        feat_comp = min(30.0, (feature_count / max(1, self.low_feature_threshold)) * 30.0)

        quality_score = round(sharp_comp + exp_comp + feat_comp, 2)
        if primary_status == "CORRUPTED":
            quality_score = 0.0

        assessment = QualityAssessment(
            status=primary_status,
            blur_score=round(blur_score, 3),
            exposure_mean=round(exposure_mean, 2),
            entropy=round(entropy, 3),
            feature_count=feature_count,
            is_corrupted=False,
            quality_score=quality_score,
            flags=flags,
        )

        return assessment

