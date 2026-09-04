"""Test 1 — OpenCV YAML parsing of the supplied DJI Air 2S calibration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from reconstruction._internal.calibration.loader import OpenCVCameraCalibrationLoader
from reconstruction._internal.models.calibration import (
    CameraCalibration,
    CameraCalibrationError,
)


# The supplied DJI Air 2S calibration in OpenCV YAML form.
# Embedded as a literal here so the test does not depend on the
# benchmark-data directory layout.
DJI_AIR_2S_YAML = """%YAML:1.0
---
camera_name: DJI_Air_2S
image_width: 1920
image_height: 1080
distortion_model: plumb_bob
camera_matrix: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ 1764.4124664880155, 0.0, 940.3814483113493,
           0.0, 1775.548789765484, 576.0927258679475,
           0.0, 0.0, 1.0 ]
distortion_coefficients: !x!opencv-matrix
   rows: 1
   cols: 5
   dt: d
   data: [ 0.08871744254107644, -1.5438032369603156,
           0.004217725112875651, -0.005798505222584687,
           5.873702095567386 ]
"""


def test_parse_dji_yaml_from_string(tmp_path: Path) -> None:
    """Parse the supplied calibration from a string and verify values."""
    path = tmp_path / "dji.yaml"
    path.write_text(DJI_AIR_2S_YAML, encoding="utf-8")

    calib: CameraCalibration = OpenCVCameraCalibrationLoader.load_from_file(path)

    assert calib.camera_name == "DJI_Air_2S"
    assert calib.image_width == 1920
    assert calib.image_height == 1080
    assert calib.distortion_model == "plumb_bob"

    # Tolerances chosen to be tight enough to catch real regressions
    # but loose enough not to be brittle against trivial floating-point
    # round-trip in the YAML loader.
    assert calib.fx == pytest.approx(1764.412466, abs=1e-6)
    assert calib.fy == pytest.approx(1775.548790, abs=1e-6)
    assert calib.cx == pytest.approx(940.381448, abs=1e-6)
    assert calib.cy == pytest.approx(576.092726, abs=1e-6)

    expected_dist = [
        0.08871744254107644,
        -1.5438032369603156,
        0.004217725112875651,
        -0.005798505222584687,
        5.873702095567386,
    ]
    assert len(calib.distortion_coefficients) == 5
    for got, exp in zip(calib.distortion_coefficients.tolist(), expected_dist):
        assert got == pytest.approx(exp, abs=1e-9)


def test_parse_dji_yaml_in_memory() -> None:
    """Same calibration parsed directly from the benchmark file."""
    repo_root = Path(__file__).resolve().parents[2]
    calib_path = repo_root / "benchmarks" / "dataset" / "video-1005" / "camera_calibration.yaml"
    if not calib_path.is_file():
        pytest.skip(f"benchmark calibration file not present at {calib_path}")
    calib = OpenCVCameraCalibrationLoader.load_from_file(calib_path)
    assert calib.camera_name == "DJI_Air_2S"
    assert calib.image_width == 1920
    assert calib.image_height == 1080


def test_load_from_dict_round_trip(tmp_path: Path) -> None:
    """Pass an in-memory dict (no file IO) through the loader."""
    data = {
        "camera_name": "SyntheticCam",
        "image_width": 800,
        "image_height": 600,
        "distortion_model": "plumb_bob",
        "camera_matrix": {
            "rows": 3,
            "cols": 3,
            "dt": "d",
            "data": [500.0, 0.0, 400.0, 0.0, 500.0, 300.0, 0.0, 0.0, 1.0],
        },
        "distortion_coefficients": {
            "rows": 1,
            "cols": 5,
            "dt": "d",
            "data": [-0.1, 0.02, 0.001, -0.002, 0.0],
        },
    }
    calib = OpenCVCameraCalibrationLoader.load_from_dict(data)
    assert calib.camera_name == "SyntheticCam"
    assert calib.fx == 500.0 and calib.fy == 500.0
    assert calib.cx == 400.0 and calib.cy == 300.0
    assert calib.distortion_coefficients.tolist() == pytest.approx(
        [-0.1, 0.02, 0.001, -0.002, 0.0]
    )
