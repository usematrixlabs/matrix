"""Unit tests for 3D Helmert 7-parameter similarity transform and RANSAC outlier rejection."""

import numpy as np
import pytest

from georeferencing_validation._internal.control_points import ControlPoints
from georeferencing_validation._internal.helmert import HelmertTransform


def test_helmert_exact_identity():
    pts = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 5.0],
    ], dtype=np.float64)

    cp = ControlPoints(source=pts, target=pts)
    transform = HelmertTransform.from_control_points(cp)

    assert pytest.approx(transform.scale, rel=1e-5) == 1.0
    np.testing.assert_allclose(transform.translation, [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(transform.rotation, np.eye(3), atol=1e-6)


def test_helmert_known_rigid_transform():
    true_scale = 2.5
    true_rot = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    true_trans = np.array([100.0, 200.0, 300.0], dtype=np.float64)

    src = np.array([
        [0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 3.0],
        [5.0, 5.0, 3.0],
    ], dtype=np.float64)

    tgt = true_scale * (src @ true_rot.T) + true_trans

    cp = ControlPoints(source=src, target=tgt)
    transform = HelmertTransform.from_control_points(cp)

    assert pytest.approx(transform.scale, rel=1e-5) == true_scale
    np.testing.assert_allclose(transform.rotation, true_rot, atol=1e-5)
    np.testing.assert_allclose(transform.translation, true_trans, atol=1e-5)

    pred = transform.transform_points(src)
    np.testing.assert_allclose(pred, tgt, atol=1e-5)


def test_helmert_ransac_outlier_rejection():
    rng = np.random.RandomState(123)
    src = rng.uniform(-10.0, 10.0, size=(8, 3))
    
    true_scale = 1.2
    true_rot = np.eye(3)
    true_trans = np.array([500.0, 600.0, 700.0])
    tgt = true_scale * (src @ true_rot.T) + true_trans

    tgt[6] += np.array([100.0, -200.0, 50.0])
    tgt[7] += np.array([-300.0, 150.0, -80.0])

    cp = ControlPoints(source=src, target=tgt)
    transform = HelmertTransform.from_control_points(
        cp,
        max_iterations=100,
        outlier_threshold=2.5,
    )

    assert transform.inlier_mask is not None
    assert transform.inlier_mask.sum() == 6
    assert transform.inlier_mask[6] == False
    assert transform.inlier_mask[7] == False
    assert pytest.approx(transform.scale, rel=1e-3) == true_scale
    np.testing.assert_allclose(transform.translation, true_trans, atol=1e-3)


def test_helmert_euler_angles_and_matrix():
    rot = np.eye(3)
    trans = np.array([1.0, 2.0, 3.0])
    transform = HelmertTransform(rotation=rot, scale=1.0, translation=trans)

    rx, ry, rz = transform.rotation_angles()
    assert pytest.approx(rx) == 0.0
    assert pytest.approx(ry) == 0.0
    assert pytest.approx(rz) == 0.0

    h_mat = transform.as_homogeneous_matrix()
    assert h_mat.shape == (4, 4)
    np.testing.assert_allclose(h_mat[:3, :3], np.eye(3))
    np.testing.assert_allclose(h_mat[:3, 3], trans)

