"""Unit tests for MultiViewTriangulator."""

import numpy as np
import pytest

from src.reconstruction.engine.triangulation import MultiViewTriangulator
from tests.fixtures.synthetic_scene import generate_synthetic_uav_dataset


def test_triangulate_synthetic_exact():
    payload, gt_points, _ = generate_synthetic_uav_dataset(
        num_frames=6, num_points=20, noise_std_px=0.0, seed=123
    )

    triangulator = MultiViewTriangulator(max_reprojection_error_px=1.0)

    # For track 0
    track_0_pts = []
    track_0_proj = []
    track_0_cams = []

    for obs in payload.observations:
        for feat in obs.features:
            if feat.track_id == "trk_0000":
                track_0_pts.append(feat.xy)
                track_0_proj.append(obs.pose.projection_matrix(obs.camera))
                r = obs.pose.rotation_matrix
                t = obs.pose.position_array
                track_0_cams.append(-r.T @ t)

    assert len(track_0_pts) >= 2

    pt_3d, mean_err = triangulator.triangulate_point_n_views(
        points_2d=np.array(track_0_pts),
        projection_matrices=track_0_proj,
        camera_centers=np.array(track_0_cams),
    )

    assert pt_3d is not None
    assert mean_err < 0.01  # Exact noiseless projection should have near-zero reprojection error
    np.testing.assert_allclose(pt_3d, gt_points[0], atol=1e-3)


def test_cheirality_rejection():
    # Point located BEHIND camera 1 (Z_cam1 = -5 < 0)
    triangulator = MultiViewTriangulator()

    # Camera 1 at origin looking along +Z
    p1 = np.array([
        [1000.0, 0.0, 500.0, 0.0],
        [0.0, 1000.0, 500.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ])
    # Camera 2 at Z=20 looking towards -Z
    p2 = np.array([
        [1000.0, 0.0, -500.0, 10000.0],
        [0.0, 1000.0, -500.0, 10000.0],
        [0.0, 0.0, -1.0, 20.0]
    ])

    # 2D points corresponding to point at [0, 0, -5]
    pts_2d = np.array([[500.0, 500.0], [500.0, 500.0]])

    pt_3d, err = triangulator.triangulate_point_n_views(
        points_2d=pts_2d,
        projection_matrices=[p1, p2],
    )

    # Must be rejected because depth < 0 in Camera 1
    assert pt_3d is None
    assert err == float("inf")



def test_insufficient_views():
    triangulator = MultiViewTriangulator()
    p1 = np.eye(3, 4)
    pt_3d, err = triangulator.triangulate_point_n_views(
        points_2d=np.array([[100.0, 100.0]]),
        projection_matrices=[p1],
    )
    assert pt_3d is None
    assert err == float("inf")
