"""Test 2 — Malformed calibration rejection."""

from __future__ import annotations

from pathlib import Path

import pytest

from reconstruction._internal.calibration.loader import OpenCVCameraCalibrationLoader
from reconstruction._internal.models.calibration import (
    CameraCalibration,
    CameraCalibrationError,
)


VALID_BASE = {
    "camera_name": "Test",
    "image_width": 1920,
    "image_height": 1080,
    "distortion_model": "plumb_bob",
    "camera_matrix": {
        "rows": 3,
        "cols": 3,
        "dt": "d",
        "data": [1000.0, 0.0, 960.0, 0.0, 1000.0, 540.0, 0.0, 0.0, 1.0],
    },
    "distortion_coefficients": {
        "rows": 1,
        "cols": 5,
        "dt": "d",
        "data": [0.0, 0.0, 0.0, 0.0, 0.0],
    },
}


def _load_with_overrides(overrides: dict) -> CameraCalibration:
    payload = {**VALID_BASE, **overrides}
    return OpenCVCameraCalibrationLoader.load_from_dict(payload)


def test_missing_camera_matrix() -> None:
    payload = {**VALID_BASE}
    payload.pop("camera_matrix")
    with pytest.raises(CameraCalibrationError, match="camera_matrix"):
        OpenCVCameraCalibrationLoader.load_from_dict(payload)


def test_wrong_matrix_dimensions() -> None:
    """A 2x2 or 4x4 matrix is rejected."""
    payload = {**VALID_BASE, "camera_matrix": {
        "rows": 2, "cols": 2, "dt": "d",
        "data": [1.0, 0.0, 0.0, 1.0],
    }}
    with pytest.raises(CameraCalibrationError, match="camera_matrix"):
        OpenCVCameraCalibrationLoader.load_from_dict(payload)

    payload = {**VALID_BASE, "camera_matrix": {
        "rows": 4, "cols": 4, "dt": "d",
        "data": [1.0] * 16,
    }}
    with pytest.raises(CameraCalibrationError, match="camera_matrix"):
        OpenCVCameraCalibrationLoader.load_from_dict(payload)


def test_negative_focal_length() -> None:
    payload = {**VALID_BASE, "camera_matrix": {
        "rows": 3, "cols": 3, "dt": "d",
        "data": [-1000.0, 0.0, 960.0, 0.0, 1000.0, 540.0, 0.0, 0.0, 1.0],
    }}
    with pytest.raises(CameraCalibrationError, match="focal lengths"):
        OpenCVCameraCalibrationLoader.load_from_dict(payload)


def test_zero_focal_length() -> None:
    payload = {**VALID_BASE, "camera_matrix": {
        "rows": 3, "cols": 3, "dt": "d",
        "data": [0.0, 0.0, 960.0, 0.0, 1000.0, 540.0, 0.0, 0.0, 1.0],
    }}
    with pytest.raises(CameraCalibrationError, match="focal lengths"):
        OpenCVCameraCalibrationLoader.load_from_dict(payload)


def test_principal_point_outside_image() -> None:
    payload = {**VALID_BASE, "camera_matrix": {
        "rows": 3, "cols": 3, "dt": "d",
        "data": [1000.0, 0.0, 5000.0, 0.0, 1000.0, 540.0, 0.0, 0.0, 1.0],
    }}
    with pytest.raises(CameraCalibrationError, match="principal point cx"):
        OpenCVCameraCalibrationLoader.load_from_dict(payload)

    payload = {**VALID_BASE, "camera_matrix": {
        "rows": 3, "cols": 3, "dt": "d",
        "data": [1000.0, 0.0, 960.0, 0.0, 1000.0, -10.0, 0.0, 0.0, 1.0],
    }}
    with pytest.raises(CameraCalibrationError, match="principal point cy"):
        OpenCVCameraCalibrationLoader.load_from_dict(payload)


def test_non_pinhole_lower_right_corner() -> None:
    """K[2,2] must be ≈ 1.0; K[2,0] and K[2,1] must be 0."""
    payload = {**VALID_BASE, "camera_matrix": {
        "rows": 3, "cols": 3, "dt": "d",
        "data": [1000.0, 0.0, 960.0, 0.0, 1000.0, 540.0, 1.0, 0.0, 2.0],
    }}
    with pytest.raises(CameraCalibrationError):
        OpenCVCameraCalibrationLoader.load_from_dict(payload)


def test_unsupported_distortion_model() -> None:
    payload = {**VALID_BASE, "distortion_model": "fisheye_equidistant"}
    with pytest.raises(CameraCalibrationError, match="Unsupported distortion_model"):
        OpenCVCameraCalibrationLoader.load_from_dict(payload)


def test_wrong_coefficient_count_plumb_bob() -> None:
    payload = {**VALID_BASE, "distortion_coefficients": {
        "rows": 1, "cols": 4, "dt": "d",
        "data": [0.1, 0.0, 0.0, 0.0],
    }}
    with pytest.raises(CameraCalibrationError, match="requires exactly 5"):
        OpenCVCameraCalibrationLoader.load_from_dict(payload)


def test_missing_distortion_coefficients() -> None:
    payload = {**VALID_BASE}
    payload.pop("distortion_coefficients")
    with pytest.raises(CameraCalibrationError, match="distortion_coefficients"):
        OpenCVCameraCalibrationLoader.load_from_dict(payload)


def test_missing_camera_name() -> None:
    payload = {**VALID_BASE}
    payload.pop("camera_name")
    with pytest.raises(CameraCalibrationError, match="camera_name"):
        OpenCVCameraCalibrationLoader.load_from_dict(payload)


def test_invalid_dimensions() -> None:
    payload = {**VALID_BASE, "image_width": 0}
    with pytest.raises(CameraCalibrationError):
        OpenCVCameraCalibrationLoader.load_from_dict(payload)
    payload = {**VALID_BASE, "image_height": -1}
    with pytest.raises(CameraCalibrationError):
        OpenCVCameraCalibrationLoader.load_from_dict(payload)


def test_file_does_not_exist(tmp_path: Path) -> None:
    with pytest.raises(CameraCalibrationError, match="does not exist"):
        OpenCVCameraCalibrationLoader.load_from_file(tmp_path / "missing.yaml")


def test_invalid_yaml(tmp_path: Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("camera_name: : :", encoding="utf-8")
    with pytest.raises(CameraCalibrationError):
        OpenCVCameraCalibrationLoader.load_from_file(path)
