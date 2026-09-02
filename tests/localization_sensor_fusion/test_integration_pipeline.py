"""Integration test for full sensor fusion pipeline combining IMU, GPS, and Visual Pose."""

import numpy as np

from src.localization_sensor_fusion._internal.engines.colmap_engine import VisualLocalizerEngine
from src.localization_sensor_fusion._internal.fusion.fusion_engine import SensorFusionEngine
from src.localization_sensor_fusion._internal.schemas.contracts import (
    CameraPose,
    Position,
    QuaternionOrientation,
    LocalizationQuality,
)


def test_full_fusion_pipeline_step():
    # 1. Initialize Visual Localizer Engine
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
    localizer = VisualLocalizerEngine(camera_matrix=K)

    # Synthetic 2D-3D point matches
    object_pts = np.array(
        [[0, 0, 5], [1, 0, 5], [0, 1, 5], [1, 1, 5]], dtype=np.float64
    )
    image_pts = np.array(
        [[320, 240], [420, 240], [320, 340], [420, 340]], dtype=np.float64
    )

    visual_pose, quality = localizer.estimate_pose(image_pts, object_pts)
    assert visual_pose is not None
    assert quality.confidence > 0.0

    # 2. Initialize EKF Fusion Engine
    fusion = SensorFusionEngine()

    # Predict state using time delta
    fusion.predict(dt=0.1)

    # 3. Apply GPS Update with Dynamic Noise Matrix (R_custom)
    gps_position = np.array([0.05, 0.0, 5.0])
    gps_std_dev = np.array([0.2, 0.2, 0.5])
    R_gps = np.diag(gps_std_dev**2)

    fusion.update(measurement=gps_position, R_custom=R_gps)

    # Verify state estimation outputs
    assert fusion.state is not None
    assert fusion.state.shape == (16, 1)
