"""Unit tests for COLMAP localization engine wrapper."""

import pytest
from src.localization_sensor_fusion.engines.colmap_engine import (
    ColmapLocalizationEngine,
)


def test_colmap_engine_valid_pose_estimation():
    engine = ColmapLocalizationEngine(min_matching_keypoints=10)
    frame_input = {
        "frame_id": "frame_001",
        "timestamp": 1.0,
        "keypoints_count": 50,
    }

    pose = engine.estimate_pose(frame_input)

    assert pose is not None
    assert pose.position.x == 0.0
    assert pose.orientation.qw == 1.0


def test_colmap_engine_insufficient_keypoints():
    engine = ColmapLocalizationEngine(min_matching_keypoints=20)
    frame_input = {
        "frame_id": "frame_002",
        "timestamp": 2.0,
        "keypoints_count": 5,
    }

    pose = engine.estimate_pose(frame_input)

    assert pose is None


def test_colmap_engine_batch_processing():
    engine = ColmapLocalizationEngine(min_matching_keypoints=10)
    batch = [
        {"frame_id": "f1", "timestamp": 1.0, "keypoints_count": 30},
        {"frame_id": "f2", "timestamp": 2.0, "keypoints_count": 2},  # Should be dropped
        {"frame_id": "f3", "timestamp": 3.0, "keypoints_count": 40},
    ]

    payload = engine.process_batch(batch)

    assert len(payload.observations) == 2
    assert payload.observations[0].localization.quality.confidence == 1.0
    assert payload.observations[0].localization.status == "estimated"