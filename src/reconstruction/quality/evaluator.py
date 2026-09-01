"""
S3 Quality Evaluator

Evaluates geometric consistency, reprojection errors, and coverage ratios,
and classifies reconstruction status (SUCCESS, WARNING, PARTIAL, FAILURE).
"""

from typing import Dict, Optional, Tuple
import numpy as np

from ..models.s3_output import ReconstructionQuality
from ..models.schema import S3Status


class QualityEvaluator:
    """Evaluates 3D reconstruction quality and determines S3 status."""

    def __init__(
        self,
        max_acceptable_mean_reproj_px: float = 2.0,
        warning_mean_reproj_px: float = 3.5,
        min_success_triangulation_ratio: float = 0.6,
        min_partial_triangulation_ratio: float = 0.2,
    ) -> None:
        """
        Initialize quality evaluator.

        Parameters:
            max_acceptable_mean_reproj_px: Maximum mean reprojection error for SUCCESS.
            warning_mean_reproj_px: Upper bound before flagging high reprojection warning.
            min_success_triangulation_ratio: Minimum fraction of tracks triangulated for SUCCESS.
            min_partial_triangulation_ratio: Minimum fraction for PARTIAL (below this is FAILURE).
        """
        self.max_acceptable_mean_reproj_px = float(max_acceptable_mean_reproj_px)
        self.warning_mean_reproj_px = float(warning_mean_reproj_px)
        self.min_success_triangulation_ratio = float(min_success_triangulation_ratio)
        self.min_partial_triangulation_ratio = float(min_partial_triangulation_ratio)

    def evaluate(
        self,
        points: np.ndarray,
        reprojection_errors: np.ndarray,
        total_observations: int,
        processed_observations: int,
        total_tracks: int,
        processing_time_s: float,
        pre_validation_status: Optional[S3Status] = None,
    ) -> Tuple[ReconstructionQuality, S3Status, Optional[str]]:
        """
        Perform quality evaluation on reconstruction results.

        Parameters:
            points: (N, 3) reconstructed points.
            reprojection_errors: (N,) reprojection errors in pixels.
            total_observations: Total number of input observations.
            processed_observations: Number of successfully ingested observations.
            total_tracks: Total candidate feature tracks.
            processing_time_s: Elapsed computation time in seconds.
            pre_validation_status: Status from input validation (if WARNING, propagated).

        Returns:
            Tuple of (ReconstructionQuality, S3Status, failure_info).
        """
        num_points = points.shape[0] if points is not None else 0

        # Base case: Zero points reconstructed
        if num_points == 0:
            quality = ReconstructionQuality(
                input_observations_count=total_observations,
                processed_observations_count=processed_observations,
                triangulated_tracks_count=0,
                triangulation_success_ratio=0.0,
                mean_reprojection_error_px=0.0,
                median_reprojection_error_px=0.0,
                coverage_ratio=0.0,
                processing_time_seconds=processing_time_s,
            )
            return quality, S3Status.FAILURE, "Zero points could be reconstructed."

        # Compute error statistics
        valid_errors = reprojection_errors[np.isfinite(reprojection_errors)]
        if len(valid_errors) > 0:
            mean_reproj = float(np.mean(valid_errors))
            median_reproj = float(np.median(valid_errors))
        else:
            mean_reproj = 0.0
            median_reproj = 0.0

        triang_ratio = (num_points / total_tracks) if total_tracks > 0 else 1.0
        coverage_ratio = (processed_observations / total_observations) if total_observations > 0 else 1.0

        quality = ReconstructionQuality(
            input_observations_count=total_observations,
            processed_observations_count=processed_observations,
            triangulated_tracks_count=num_points,
            triangulation_success_ratio=triang_ratio,
            mean_reprojection_error_px=mean_reproj,
            median_reprojection_error_px=median_reproj,
            coverage_ratio=coverage_ratio,
            processing_time_seconds=processing_time_s,
        )

        # Status Classification
        status = S3Status.SUCCESS
        failure_info = None

        if triang_ratio < self.min_partial_triangulation_ratio:
            status = S3Status.FAILURE
            failure_info = f"Critical low triangulation ratio ({triang_ratio:.2%})"
        elif triang_ratio < self.min_success_triangulation_ratio:
            status = S3Status.PARTIAL
            failure_info = f"Partial reconstruction achieved ({triang_ratio:.2%} tracks)"
        elif mean_reproj > self.max_acceptable_mean_reproj_px or pre_validation_status == S3Status.WARNING:
            status = S3Status.WARNING
            if mean_reproj > self.max_acceptable_mean_reproj_px:
                failure_info = f"High mean reprojection error ({mean_reproj:.2f} px)"

        return quality, status, failure_info

