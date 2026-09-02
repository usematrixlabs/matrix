"""Unit tests for S4 Spatial Consistency evaluation."""

import numpy as np
import pytest

from src.georeferencing_validation._internal.control_points import ControlPoints
from src.georeferencing_validation._internal.helmert import HelmertTransform
from src.georeferencing_validation._internal.validator import GeoreferencingValidator


def test_spatial_consistency_planar_grid():
    xx, yy = np.meshgrid(np.linspace(0, 10, 11), np.linspace(0, 10, 11))
    src_points = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(121)])

    cp_src = src_points[[0, 1, 11, 12]]
    cp_tgt = cp_src + np.array([500000.0, 3000000.0, 100.0])
    cp = ControlPoints(source=cp_src, target=cp_tgt)
    transform = HelmertTransform(
        rotation=np.eye(3),
        scale=1.0,
        translation=np.array([500000.0, 3000000.0, 100.0]),
    )

    validator = GeoreferencingValidator(control_points=cp, transformation=transform)
    tgt_points = transform.transform_points(src_points)

    report = validator.check_spatial_consistency(
        points=tgt_points,
        source_points=src_points,
    )

    assert report.passed is True
    assert report.spatial_consistency_score > 0.6
    assert pytest.approx(report.plane_fit_residual_rmse, abs=1e-5) == 0.0
    assert report.scale_preservation_max_error < 1e-4
    assert len(report.warnings) == 0


def test_spatial_consistency_empty_or_small():
    cp = ControlPoints(
        source=np.array([[0,0,0],[1,0,0],[0,1,0]], dtype=np.float64),
        target=np.array([[0,0,0],[1,0,0],[0,1,0]], dtype=np.float64),
    )
    transform = HelmertTransform(rotation=np.eye(3), scale=1.0, translation=np.zeros(3))
    validator = GeoreferencingValidator(control_points=cp, transformation=transform)

    small_pts = np.array([[0,0,0], [1,1,1]], dtype=np.float64)
    report = validator.check_spatial_consistency(small_pts)
    assert report.passed is True
    assert report.spatial_consistency_score == 1.0

