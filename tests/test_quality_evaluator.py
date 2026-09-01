"""Unit tests for S3 Quality Evaluator."""

import numpy as np
import pytest

from src.reconstruction.models.schema import S3Status
from src.reconstruction.quality.evaluator import QualityEvaluator


def test_evaluate_zero_points():
    evaluator = QualityEvaluator()
    quality, status, failure_info = evaluator.evaluate(
        points=np.empty((0, 3)),
        reprojection_errors=np.empty((0,)),
        total_observations=5,
        processed_observations=5,
        total_tracks=20,
        processing_time_s=0.5,
    )
    assert status == S3Status.FAILURE
    assert quality.triangulated_tracks_count == 0
    assert "Zero points" in failure_info


def test_evaluate_high_quality_success():
    evaluator = QualityEvaluator(max_acceptable_mean_reproj_px=2.0)
    pts = np.ones((50, 3))
    errs = np.full(50, 0.65)  # 0.65px reprojection error
    quality, status, failure_info = evaluator.evaluate(
        points=pts,
        reprojection_errors=errs,
        total_observations=10,
        processed_observations=10,
        total_tracks=55,
        processing_time_s=1.2,
    )
    assert status == S3Status.SUCCESS
    assert failure_info is None
    assert quality.mean_reprojection_error_px == 0.65
    assert quality.triangulation_success_ratio > 0.9


def test_evaluate_high_reprojection_warning():
    evaluator = QualityEvaluator(max_acceptable_mean_reproj_px=2.0)
    pts = np.ones((30, 3))
    errs = np.full(30, 2.8)  # 2.8px error > 2.0px
    quality, status, failure_info = evaluator.evaluate(
        points=pts,
        reprojection_errors=errs,
        total_observations=5,
        processed_observations=5,
        total_tracks=32,
        processing_time_s=0.8,
    )
    assert status == S3Status.WARNING
    assert "High mean reprojection error" in failure_info


def test_evaluate_low_triangulation_partial():
    evaluator = QualityEvaluator(min_success_triangulation_ratio=0.6, min_partial_triangulation_ratio=0.2)
    pts = np.ones((10, 3))
    errs = np.full(10, 0.8)
    # 10 points out of 30 tracks = 33% (between 20% and 60%)
    quality, status, failure_info = evaluator.evaluate(
        points=pts,
        reprojection_errors=errs,
        total_observations=5,
        processed_observations=5,
        total_tracks=30,
        processing_time_s=0.5,
    )
    assert status == S3Status.PARTIAL
    assert "Partial reconstruction" in failure_info

