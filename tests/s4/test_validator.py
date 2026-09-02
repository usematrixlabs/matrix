"""Unit tests for S4 Accuracy Validator and horizontal/vertical error splits."""

import numpy as np
import pytest

from src.georeferencing_validation.control_points import ControlPoints
from src.georeferencing_validation.helmert import HelmertTransform
from src.georeferencing_validation.validator import GeoreferencingValidator


def test_validator_metrics_exact_zero():
    pts = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 5.0],
    ], dtype=np.float64)

    cp = ControlPoints(source=pts, target=pts)
    transform = HelmertTransform(rotation=np.eye(3), scale=1.0, translation=np.zeros(3))

    validator = GeoreferencingValidator(
        control_points=cp,
        transformation=transform,
        tolerance=0.1,
        horizontal_tolerance=0.05,
        vertical_tolerance=0.05,
    )
    res = validator.validate()

    assert pytest.approx(res.rmse) == 0.0
    assert pytest.approx(res.horizontal_rmse) == 0.0
    assert pytest.approx(res.vertical_rmse) == 0.0
    assert res.passed_3d is True
    assert res.passed_horizontal is True
    assert res.passed_vertical is True
    assert res.passed is True


def test_validator_horizontal_vs_vertical_split():
    src = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [10.0, 10.0, 0.0],
    ], dtype=np.float64)

    tgt = src + np.array([0.0, 0.0, 0.8])
    cp = ControlPoints(source=src, target=tgt)

    transform = HelmertTransform(rotation=np.eye(3), scale=1.0, translation=np.zeros(3))

    validator = GeoreferencingValidator(
        control_points=cp,
        transformation=transform,
        horizontal_tolerance=0.1,
        vertical_tolerance=0.5,
    )
    res = validator.validate()

    assert pytest.approx(res.horizontal_rmse) == 0.0
    assert pytest.approx(res.vertical_rmse) == 0.8
    assert pytest.approx(res.rmse) == 0.8

    assert res.passed_horizontal is True
    assert res.passed_vertical is False
    assert res.passed is False


def test_validator_to_dict_and_tolerances():
    src = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)
    cp = ControlPoints(source=src, target=src)
    transform = HelmertTransform(rotation=np.eye(3), scale=1.0, translation=np.zeros(3))

    validator = GeoreferencingValidator(control_points=cp, transformation=transform)
    res = validator.validate()

    d = res.to_dict()
    assert "metrics" in d
    assert "rmse_3d" in d["metrics"]
    assert "horizontal_rmse" in d["metrics"]
    assert "vertical_rmse" in d["metrics"]
    assert d["pass_status"]["overall_passed"] is None

