"""Unit tests for S4 Ground Control Point (GCP) management."""

import numpy as np
import pytest

from georeferencing_validation._internal.control_points import ControlPoints


def test_control_points_valid():
    src = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 5.0],
    ], dtype=np.float64)
    tgt = src + np.array([500000.0, 3000000.0, 100.0])

    cp = ControlPoints(source=src, target=tgt, metadata={"survey_date": "2026-09-01"})
    assert cp.number_of_points == 4
    np.testing.assert_allclose(cp.source_centroid, [2.5, 2.5, 1.25])
    assert cp.metadata["survey_date"] == "2026-09-01"


def test_control_points_insufficient_points():
    src = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    tgt = np.array([[10.0, 10.0, 10.0], [11.0, 11.0, 11.0]])
    with pytest.raises(ValueError, match="At least 3 corresponding control points"):
        ControlPoints(source=src, target=tgt)


def test_control_points_collinear_degeneracy():
    src = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
    ], dtype=np.float64)
    tgt = np.array([
        [10.0, 10.0, 10.0],
        [11.0, 11.0, 11.0],
        [12.0, 12.0, 12.0],
    ], dtype=np.float64)

    with pytest.raises(ValueError, match="geometrically degenerate"):
        ControlPoints(source=src, target=tgt)


def test_control_points_duplicates():
    src = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)
    tgt = np.array([
        [10.0, 10.0, 10.0],
        [10.0, 10.0, 10.0],
        [10.0, 11.0, 10.0],
    ], dtype=np.float64)

    with pytest.raises(ValueError, match="Duplicate points detected"):
        ControlPoints(source=src, target=tgt)


def test_control_points_nan_inf():
    src = np.array([[0.0, 0.0, 0.0], [1.0, np.nan, 0.0], [0.0, 1.0, 0.0]])
    tgt = np.array([[10.0, 10.0, 10.0], [11.0, 10.0, 10.0], [10.0, 11.0, 10.0]])
    with pytest.raises(ValueError, match="finite numerical values"):
        ControlPoints(source=src, target=tgt)

