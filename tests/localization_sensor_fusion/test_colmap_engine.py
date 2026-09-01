import numpy as np
import pytest
from src.localization_sensor_fusion.engines.colmap_engine import VisualLocalizerEngine

def test_pnp_pose_estimation_success():
    K = np.array([
        [500.0, 0.0, 320.0],
        [0.0, 500.0, 240.0],
        [0.0, 0.0, 1.0]
    ])
    engine = VisualLocalizerEngine(camera_matrix=K)

    # 4 synthetic 3D points
    object_pts = np.array([
        [-1.0, -1.0, 2.0],
        [1.0, -1.0, 2.0],
        [1.0, 1.0, 2.0],
        [-1.0, 1.0, 2.0]
    ], dtype=np.float64)

    # Project points to 2D image plane with identity rotation and z=2 translation
    image_pts = np.array([
        [70.0, 0.0],
        [570.0, 0.0],
        [570.0, 490.0],
        [70.0, 490.0]
    ], dtype=np.float64)

    pose, quality = engine.estimate_pose(image_pts, object_pts)

    assert pose is not None
    assert quality.confidence > 0.0
    assert isinstance(pose.position.x, float)

def test_pnp_pose_insufficient_points():
    K = np.eye(3)
    engine = VisualLocalizerEngine(camera_matrix=K)
    
    pose, quality = engine.estimate_pose(np.zeros((2, 2)), np.zeros((2, 3)))
    assert pose is None
    assert quality.confidence == 0.0

def test_orb_feature_extraction_and_matching():
    K = np.eye(3)
    engine = VisualLocalizerEngine(camera_matrix=K)

    # Create synthetic 100x100 grayscale image frame with a white square
    frame = np.zeros((100, 100), dtype=np.uint8)
    frame[20:80, 20:80] = 255

    # Test feature extraction on empty map
    img_pts, obj_pts = engine.extract_and_match_features(frame, map_descriptors=None, map_3d_points=None)
    assert len(img_pts) == 0
    assert len(obj_pts) == 0

    # Test pose from frame with empty input handling
    pose, quality = engine.estimate_pose_from_frame(frame, map_descriptors=None, map_3d_points=None)
    assert pose is None
    assert quality.confidence == 0.0