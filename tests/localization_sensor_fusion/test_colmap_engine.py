"""Unit tests for ColmapLocalizationEngine."""

from src.localization_sensor_fusion.engines.colmap_engine import (
    ColmapLocalizationEngine,
)


def test_colmap_engine_valid_keypoints_and_pose():
    engine = ColmapLocalizationEngine(min_keypoints_threshold=10)

    input_data = {
        "frame_id": "frame_001",
        "image_path": "path/to/img.jpg",
        "keypoints_count": 25,
        "timestamp": 123.456,
        "computed_pose": {
            "position": {"x": 1.2, "y": 3.4, "z": 5.6},
            "orientation": {"qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0},
        },
    }

    results = engine.process_batch([input_data])

    assert len(results) == 1
    obs = results[0]
    assert obs.observation_id == "frame_001"
    assert obs.image == "path/to/img.jpg"
    assert obs.pose is not None
    assert obs.pose.position.x == 1.2


def test_colmap_engine_missing_pose_returns_empty():
    engine = ColmapLocalizationEngine(min_keypoints_threshold=10)

    # Keypoints passed threshold, but no pose was computed by COLMAP
    input_data = {
        "frame_id": "frame_002",
        "image_path": "path/to/img2.jpg",
        "keypoints_count": 25,
        "timestamp": 123.456,
    }

    results = engine.process_batch([input_data])
    assert len(results) == 0


def test_colmap_engine_insufficient_keypoints():
    engine = ColmapLocalizationEngine(min_keypoints_threshold=10)

    input_data = {
        "frame_id": "frame_003",
        "image_path": "path/to/img3.jpg",
        "keypoints_count": 5,
        "computed_pose": {
            "position": {"x": 1.0, "y": 1.0, "z": 1.0},
            "orientation": {"qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0},
        },
        "timestamp": 123.456,
    }

    results = engine.process_batch([input_data])
    assert len(results) == 0