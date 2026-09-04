"""Test 4 — Undistortion correctness.

We verify that ``cv2.undistortPoints`` with ``P=K`` returns
*undistorted pixel coordinates in the same K pixel space*:

* The principal point and image center remain essentially unchanged
  (distortion is zero at the principal point by definition).
* The inverse relationship is checked: a point that is *undistorted*
  and then re-distorted by ``cv2.projectPoints`` round-trips back to
  the original distorted observation.
"""

from __future__ import annotations

import numpy as np
import pytest

from reconstruction._internal.calibration.loader import OpenCVCameraCalibrationLoader
from reconstruction._internal.preprocessing.undistort import ObservationUndistorter


@pytest.fixture
def dji_calib():
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    calib_path = repo_root / "benchmarks" / "dataset" / "video-1005" / "camera_calibration.yaml"
    if not calib_path.is_file():
        pytest.skip(f"benchmark calibration file not present at {calib_path}")
    return OpenCVCameraCalibrationLoader.load_from_file(calib_path)


def test_principal_point_unchanged(dji_calib) -> None:
    """A point at the principal point should be essentially untouched."""
    pts = np.array([[dji_calib.cx, dji_calib.cy]], dtype=np.float64)
    result = ObservationUndistorter.undistort_points(pts, dji_calib)
    np.testing.assert_allclose(
        result.undistorted_uv, pts, atol=1e-3,
        err_msg="principal point must not move under undistortion",
    )


def test_zero_distortion_passthrough() -> None:
    """When D is all zeros the result equals the input."""
    import numpy as np
    from reconstruction._internal.models.calibration import CameraCalibration

    calib = CameraCalibration(
        camera_name="ZeroDist",
        image_width=1920,
        image_height=1080,
        distortion_model="plumb_bob",
        camera_matrix=np.array([
            [1000.0, 0.0, 960.0],
            [0.0, 1000.0, 540.0],
            [0.0, 0.0, 1.0],
        ]),
        distortion_coefficients=np.zeros(5, dtype=np.float64),
    )
    pts = np.array([
        [100.0, 200.0],
        [1820.0, 980.0],
        [960.0, 540.0],
    ], dtype=np.float64)
    result = ObservationUndistorter.undistort_points(pts, calib)
    np.testing.assert_allclose(result.undistorted_uv, pts, atol=1e-9)
    assert result.applied is False


def test_round_trip_via_project_points(dji_calib) -> None:
    """``cv2.projectPoints`` should invert our undistortion within tolerance."""
    import cv2

    # Use a known distorted point: pick an off-center pixel and pretend
    # it is what the detector observed.
    pts = np.array([
        [100.0, 100.0],
        [1820.0, 100.0],
        [1820.0, 980.0],
        [100.0, 980.0],
        [960.0, 540.0],
    ], dtype=np.float64)

    K = dji_calib.camera_matrix.astype(np.float64)
    D = dji_calib.distortion_coefficients.reshape(-1, 1).astype(np.float64)

    # 1) Undistort
    und = cv2.undistortPoints(
        pts.reshape(-1, 1, 2),
        K,
        D,
        P=K,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-6),
    ).reshape(-1, 2)

    # 2) Re-project (forward model) — but cv2.projectPoints expects
    #    object-space points. Build a trivial 3D point per pixel by
    #    using the undistorted pixel as (u, v, depth=1) in K-space:
    #    X_cam = K^{-1} (u, v, 1), then project.
    Kinv = np.linalg.inv(K)
    homog = np.hstack([und, np.ones((und.shape[0], 1))])
    X_cam = (Kinv @ homog.T).T  # (N, 3)
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    reproj, _ = cv2.projectPoints(X_cam, rvec, tvec, K, D)
    reproj = reproj.reshape(-1, 2)

    np.testing.assert_allclose(reproj, pts, atol=0.5, err_msg="round trip should recover distorted pixel")
