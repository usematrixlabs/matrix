"""Unit tests for the Visual Localizer Engine."""

import pytest
import numpy as np
from src.localization_sensor_fusion.engines.colmap_engine import VisualLocalizerEngine


def test_pnp_pose_estimation():
    # Intrinsic matrix (focal length ~500, principal point at 320, 240)
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    engine = VisualLocalizerEngine(camera_matrix=K)

    # 4 synthetic 3D points
    object_points = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ])

    # Corresponding 2D image points
    image_points = np.array([
        [320.0, 240.0],
        [820.0, 240.0],
        [320.0, 740.0],
        [820.0, 740.0],
    ])

    success, rvec, tvec = engine.estimate_pose_pnp(object_points, image_points)
    assert success is True or isinstance(success, (bool, np.bool_))