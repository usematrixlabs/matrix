"""Test 5 — Synthetic calibrated triangulation.

End-to-end verification that:

1. A 3D point projected through ``cv2.projectPoints(K, D)`` produces a
   *distorted* pixel observation.
2. The Matrix preprocessing pipeline recovers an *undistorted* pixel
   coordinate via ``cv2.undistortPoints(P=K)``.
3. The :class:`MultiViewTriangulator` uses the undistorted pixel
   coordinates (consistent with ``P = K [R | t]``) and recovers the
   ground-truth 3D point.

This is the critical regression test for the entire
``calibration → undistortion → triangulation`` chain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import pytest

from reconstruction._internal.models.calibration import CameraCalibration
from reconstruction._internal.models.schema import (
    CameraIntrinsics,
    CameraPose,
    FeatureObservation,
    LocalizationInfo,
    S2Observation,
    S2Payload,
)
from reconstruction._internal.preprocessing.prepare import ReconstructionDataPreparer
from reconstruction._internal.preprocessing.undistort import ObservationUndistorter
from reconstruction._internal.engine.triangulation import MultiViewTriangulator


# ---------------------------------------------------------------------------
# Synthetic scene helpers
# ---------------------------------------------------------------------------

@dataclass
class _Camera:
    """A calibrated pinhole camera in 3D world space."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion: np.ndarray  # (N,)
    r_mat: np.ndarray  # (3, 3) world-to-camera rotation
    t_vec: np.ndarray  # (3,) world-to-camera translation
    K: np.ndarray  # (3, 3)

    def project(self, X_world: np.ndarray) -> np.ndarray:
        """Project (N, 3) world-space points to distorted pixel coords (N, 2)."""
        X_cam = (self.r_mat @ X_world.T).T + self.t_vec  # (N, 3)
        # Cheirality: drop points behind camera
        if X_cam.shape[0] == 0:
            return np.zeros((0, 2), dtype=np.float64)
        proj, _ = cv2.projectPoints(
            X_cam.reshape(-1, 1, 3),
            np.zeros((3, 1), dtype=np.float64),
            np.zeros((3, 1), dtype=np.float64),
            self.K,
            self.distortion.reshape(-1, 1).astype(np.float64),
        )
        return proj.reshape(-1, 2)


def _build_calibration() -> CameraCalibration:
    """Build a moderate-strength synthetic calibration (1920x1080)."""
    K = np.array([
        [1200.0, 0.0, 960.0],
        [0.0, 1200.0, 540.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    D = np.array([0.15, -0.4, 0.001, -0.002, 0.1], dtype=np.float64)
    return CameraCalibration(
        camera_name="Synthetic",
        image_width=1920,
        image_height=1080,
        distortion_model="plumb_bob",
        camera_matrix=K,
        distortion_coefficients=D,
    )


def _build_cameras(num_frames: int, calib: CameraCalibration) -> List[_Camera]:
    """Build a small UAV-style downward-looking flight over the scene."""
    rng = np.random.RandomState(7)
    cams: List[_Camera] = []
    start_x = -3.0
    end_x = 3.0
    xs = np.linspace(start_x, end_x, num_frames)
    # Camera looking straight down: world +Y maps to camera -Z (forward).
    R_down = np.array([
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ], dtype=np.float64)
    for x in xs:
        t = -R_down @ np.array([x, 0.0, 10.0], dtype=np.float64)
        cams.append(_Camera(
            fx=calib.fx, fy=calib.fy, cx=calib.cx, cy=calib.cy,
            width=calib.image_width, height=calib.image_height,
            distortion=calib.distortion_coefficients,
            r_mat=R_down, t_vec=t, K=calib.camera_matrix,
        ))
    return cams


def _make_s2_payload(
    cams: List[_Camera],
    gt_points: np.ndarray,
    calib: CameraCalibration,
) -> S2Payload:
    """Generate an S2Payload by projecting gt_points into every camera."""
    observations: List[S2Observation] = []
    intrinsics = CameraIntrinsics(
        fx=calib.fx, fy=calib.fy, cx=calib.cx, cy=calib.cy,
        width=calib.image_width, height=calib.image_height,
        distortion_coefficients=calib.distortion_coefficients.tolist(),
        distortion_model=calib.distortion_model,
    )
    for i, cam in enumerate(cams):
        feats: List[FeatureObservation] = []
        for pt_idx, X in enumerate(gt_points):
            uv_dist = cam.project(X.reshape(1, 3))
            u, v = uv_dist[0]
            if not (0 <= u < cam.width and 0 <= v < cam.height):
                continue
            feats.append(FeatureObservation(
                feature_id=f"f_{i}_{pt_idx}",
                xy=(float(u), float(v)),
                track_id=f"trk_{pt_idx:04d}",
                rgb=(128, 128, 128),
            ))
        pose = CameraPose(
            position=cam.t_vec.tolist(),
            orientation=cam.r_mat.tolist(),
            orientation_format="ROTATION_MATRIX",
        )
        observations.append(S2Observation(
            observation_id=f"frame_{i:04d}",
            timestamp=float(i),
            image_path=f"frame_{i:04d}.jpg",
            camera=intrinsics,
            pose=pose,
            localization=LocalizationInfo(status="ESTIMATED", confidence=0.9),
            features=feats,
        ))
    return S2Payload(
        observations=observations,
        job_id="synthetic_calibrated_test",
        coordinate_frame="S2_LOCAL",
        units="meters",
    )


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------

def test_synthetic_calibrated_triangulation_recovers_ground_truth() -> None:
    """End-to-end: project distorted → preprocess → triangulate → match GT."""
    calib = _build_calibration()
    rng = np.random.RandomState(123)
    num_points = 25
    gt_points = np.zeros((num_points, 3), dtype=np.float64)
    gt_points[:, 0] = rng.uniform(-2.0, 2.0, size=num_points)
    gt_points[:, 1] = rng.uniform(-2.0, 2.0, size=num_points)
    gt_points[:, 2] = 0.0  # on the ground plane

    cams = _build_cameras(num_frames=6, calib=calib)
    payload = _make_s2_payload(cams, gt_points, calib)

    preparer = ReconstructionDataPreparer(calibration=calib)
    prepared = preparer.prepare(payload)

    assert prepared.metadata.get("camera_calibration") is not None
    assert prepared.metadata.get("undistortion_applied") is True

    # The preparer should populate both raw and undistorted for each track.
    tracks = prepared.tracks
    assert len(tracks) >= 5
    for tr in tracks:
        assert tr.points_2d_raw is not None
        assert tr.points_2d.shape == tr.points_2d_raw.shape
        # Raw and undistorted should differ for at least some observations
        # (this calibration has non-zero distortion everywhere except the
        # principal point).
        assert not np.allclose(tr.points_2d, tr.points_2d_raw)

    # Triangulate every track with relaxed reprojection threshold.
    triangulator = MultiViewTriangulator(max_reprojection_error_px=3.0)
    recovered: List[Tuple[int, np.ndarray]] = []
    for tr in tracks:
        pt_3d, _ = triangulator.triangulate_point_n_views(
            points_2d=tr.points_2d,
            projection_matrices=tr.projection_matrices,
            camera_centers=tr.camera_centers,
        )
        if pt_3d is None:
            continue
        # Recover the GT index from the track id
        idx = int(tr.track_id.split("_")[1])
        recovered.append((idx, pt_3d))

    assert len(recovered) >= 5
    for idx, pt_3d in recovered:
        np.testing.assert_allclose(
            pt_3d, gt_points[idx], atol=0.05,
            err_msg=f"recovered point {idx} differs from ground truth",
        )


def test_no_calibration_baseline_still_works() -> None:
    """Without calibration the pipeline still runs (using raw points)."""
    calib = _build_calibration()
    rng = np.random.RandomState(123)
    gt_points = np.zeros((10, 3), dtype=np.float64)
    gt_points[:, 0] = rng.uniform(-2.0, 2.0, size=10)
    gt_points[:, 1] = rng.uniform(-2.0, 2.0, size=10)
    gt_points[:, 2] = 0.0
    cams = _build_cameras(num_frames=4, calib=calib)
    payload = _make_s2_payload(cams, gt_points, calib)

    preparer = ReconstructionDataPreparer()  # no calibration
    prepared = preparer.prepare(payload)
    assert "camera_calibration" not in prepared.metadata
    for tr in prepared.tracks:
        assert tr.undistortion_applied is False
        # When no calibration, points_2d == points_2d_raw
        if tr.points_2d_raw is not None:
            np.testing.assert_allclose(tr.points_2d, tr.points_2d_raw)
