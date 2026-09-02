"""Unit tests for S1 Input Adapter."""

import pytest
from src.localization_sensor_fusion._internal.adapters.s1_adapter import (
    S1AdapterValidationError,
    S1InputAdapter,
)


def test_s1_adapter_transform_valid():
    adapter = S1InputAdapter(min_blur_score=0.5)
    payload = {
        "observation_id": "frame_100",
        "timestamp": 12.34,
        "image": "data/frames/100.png",
        "quality": {"status": "GOOD", "blur_score": 0.88},
    }
    obs = adapter.parse_observation(payload)
    engine_data = adapter.transform_for_engine(obs)

    assert engine_data is not None
    assert engine_data["frame_id"] == "frame_100"
    assert engine_data["image_path"] == "data/frames/100.png"


def test_s1_adapter_rejects_low_quality():
    adapter = S1InputAdapter(min_blur_score=0.5)
    payload = {
        "observation_id": "frame_101",
        "timestamp": 12.35,
        "image": "data/frames/101.png",
        "quality": {"status": "BLURRY", "blur_score": 0.21},
    }
    obs = adapter.parse_observation(payload)
    engine_data = adapter.transform_for_engine(obs)

    assert engine_data is None


def test_s1_adapter_invalid_schema():
    adapter = S1InputAdapter()
    invalid_payload = {"timestamp": "invalid_timestamp_type"}

    with pytest.raises(S1AdapterValidationError):
        adapter.parse_observation(invalid_payload)