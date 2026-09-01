"""Unit tests for SensorFusionEngine."""

from src.localization_sensor_fusion.fusion.fusion_engine import (
    SensorFusionEngine,
)
from src.localization_sensor_fusion.schemas.contracts import (
    CameraPose,
    Position,
    S2ObservationOutput,
    S2PayloadOutput,
)


def test_fusion_engine_weighted_pose_processing():
    engine = SensorFusionEngine(min_confidence_threshold=0.5)

    obs1 = S2ObservationOutput(
        observation_id="obs_1",
        timestamp=100.0,
        image="img_1",
        pose=CameraPose(
            position={"x": 1.0, "y": 0.0, "z": 0.0},
            orientation={"qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0},
        ),
        localization={
            "status": "estimated",
            "source": ["visual"],
            "quality": {"confidence": 0.8},
        },
    )

    obs2 = S2ObservationOutput(
        observation_id="obs_2",
        timestamp=100.1,
        image="img_2",
        pose=CameraPose(
            position={"x": 2.0, "y": 0.0, "z": 0.0},
            orientation={"qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0},
        ),
        localization={
            "status": "estimated",
            "source": ["visual"],
            "quality": {"confidence": 0.2},  # Filtered out (< 0.5)
        },
    )

    payload = S2PayloadOutput(
        session_id="session_test", observations=[obs1, obs2]
    )

    fused_pose = engine.process_payload(payload)

    assert fused_pose is not None
    assert fused_pose.position.x == 1.0


def test_fusion_engine_empty_or_low_confidence_payload():
    engine = SensorFusionEngine(min_confidence_threshold=0.8)

    obs_low = S2ObservationOutput(
        observation_id="obs_low",
        timestamp=100.0,
        image="img_low",
        pose=CameraPose(
            position={"x": 10.0, "y": 10.0, "z": 10.0},
            orientation={"qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0},
        ),
        localization={
            "status": "estimated",
            "source": ["visual"],
            "quality": {"confidence": 0.3},
        },
    )

    payload = S2PayloadOutput(
        session_id="session_low", observations=[obs_low]
    )

    assert engine.process_payload(payload) is None