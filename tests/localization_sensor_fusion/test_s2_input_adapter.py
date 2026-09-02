"""Unit tests for S2InputAdapter."""

import pytest
from src.localization_sensor_fusion._internal.adapters.s2_input_adapter import (
    S2InputAdapter,
)


def test_s2_input_adapter_valid_transform():
    adapter = S2InputAdapter()
    raw_data = {
        "frame_id": "frame_100",
        "timestamp": 123.456,
        "pose": {
            "position": {"x": 1.0, "y": 2.0, "z": 3.0},
            "orientation": {"qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0},
        },
        "confidence": 0.95,
    }

    observation = adapter.adapt(raw_data)

    assert observation.observation_id == "frame_100"
    assert observation.timestamp == 123.456
    assert observation.localization.status == "estimated"
    assert observation.localization.source == ["visual"]
    assert observation.localization.quality.confidence == 0.95


def test_s2_input_adapter_missing_fields_raises_error():
    adapter = S2InputAdapter()
    invalid_raw_data = {"frame_id": "frame_101"}  # Missing pose and timestamp

    with pytest.raises(ValueError):
        adapter.adapt(invalid_raw_data)