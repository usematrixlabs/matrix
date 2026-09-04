"""Test 3 — Image-size compatibility policy.

Verify that:
* matching resolutions are accepted directly,
* a different resolution is rejected by the pipeline (no silent reuse),
* the explicit ``scale_to_resolution`` policy scales K correctly when
  the resize is isotropic.
"""

from __future__ import annotations

import numpy as np
import pytest

from reconstruction._internal.models.calibration import (
    CameraCalibration,
    CameraCalibrationError,
)


def _make_calib(w: int = 1920, h: int = 1080) -> CameraCalibration:
    return CameraCalibration(
        camera_name="Test",
        image_width=w,
        image_height=h,
        distortion_model="plumb_bob",
        camera_matrix=np.array(
            [
                [1000.0, 0.0, w / 2.0],
                [0.0, 1000.0, h / 2.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        distortion_coefficients=np.array([0.05, -0.1, 0.0, 0.0, 0.0], dtype=np.float64),
    )


def test_matching_resolution_accepted() -> None:
    calib = _make_calib()
    scaled = calib.scale_to_resolution(1920, 1080)
    assert (scaled.image_width, scaled.image_height) == (1920, 1080)
    # fx/fy/cx/cy unchanged
    assert scaled.fx == pytest.approx(calib.fx)
    assert scaled.fy == pytest.approx(calib.fy)
    assert scaled.cx == pytest.approx(calib.cx)
    assert scaled.cy == pytest.approx(calib.cy)
    # distortion coefficients unchanged
    np.testing.assert_allclose(scaled.distortion_coefficients, calib.distortion_coefficients)


def test_isotropic_downscale() -> None:
    calib = _make_calib(1920, 1080)
    scaled = calib.scale_to_resolution(960, 540)
    # sx = sy = 0.5
    assert scaled.fx == pytest.approx(500.0)
    assert scaled.fy == pytest.approx(500.0)
    assert scaled.cx == pytest.approx(480.0)
    assert scaled.cy == pytest.approx(270.0)
    np.testing.assert_allclose(scaled.distortion_coefficients, calib.distortion_coefficients)


def test_isotropic_upscale() -> None:
    calib = _make_calib(640, 480)
    scaled = calib.scale_to_resolution(1280, 960)
    assert scaled.fx == pytest.approx(2000.0)
    assert scaled.fy == pytest.approx(2000.0)
    # cx/cy scale by sx=2, sy=2 from the 640/480-centred principal point.
    assert scaled.cx == pytest.approx(640.0)
    assert scaled.cy == pytest.approx(480.0)


def test_anisotropic_resize_rejected() -> None:
    """Non-uniform scaling is rejected (would distort the calibration)."""
    calib = _make_calib(1920, 1080)
    with pytest.raises(CameraCalibrationError, match="cannot be uniformly scaled"):
        calib.scale_to_resolution(1920, 540)  # half height only


def test_scale_preserves_camera_name_and_model() -> None:
    calib = _make_calib()
    scaled = calib.scale_to_resolution(960, 540)
    assert scaled.camera_name == calib.camera_name
    assert scaled.distortion_model == calib.distortion_model
    assert scaled.extra.get("scaled_from_resolution") == [1920, 1080]
