from src.localization_sensor_fusion.schemas.contracts import (
    S1ObservationInput,
    QualityStatus,
)


def test_s1_observation_input_valid():
    payload = {
        "observation_id": "frame_001",
        "timestamp": 1.23,
        "image": "path/to/frame.jpg",
        "quality": {"status": "GOOD", "blur_score": 0.95},
    }
    obs = S1ObservationInput(**payload)
    assert obs.observation_id == "frame_001"
    assert obs.quality is not None
    assert obs.quality.status == QualityStatus.GOOD