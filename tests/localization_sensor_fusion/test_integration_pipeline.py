"""End-to-End Integration Test for S2 Localization & Sensor Fusion Pipeline."""

import json
from pathlib import Path
from src.localization_sensor_fusion.fusion.fusion_engine import SensorFusionEngine
from src.localization_sensor_fusion.engines.trajectory_smoother import TrajectorySmoother
from src.localization_sensor_fusion.exporters.s2_exporter import S2Exporter
from src.localization_sensor_fusion.schemas.contracts import (
    S2ObservationOutput,
    CameraPose,
    Position,
    QuaternionOrientation,
    LocalizationQuality,
    LocalizationMeta,
)


def test_full_s2_pipeline_integration(tmp_path: Path):
    # 1. Initialize Engines & Exporter
    fusion_engine = SensorFusionEngine()
    smoother = TrajectorySmoother(window_size=3)
    exporter = S2Exporter(coordinate_frame="NED", units="meters")

    # 2. Build mock observations simulating a short sequence of frames
    raw_observations = []

    for i in range(5):
        obs = S2ObservationOutput(
            observation_id=f"frame_00{i}",
            timestamp=float(i * 0.1),
            image=f"frame_00{i}.jpg",
            pose=CameraPose(
                position=Position(x=float(i), y=float(i * 2), z=0.5),
                orientation=QuaternionOrientation(qw=1.0, qx=0.0, qy=0.0, qz=0.0),
            ),
            localization=LocalizationMeta(
                source=["visual"],
                status="estimated",
                quality=LocalizationQuality(confidence=0.90),
            ),
        )
        raw_observations.append(obs)

    # 3. Step 1: Pass through Sensor Fusion Engine (EKF)
    fused_obs = fusion_engine.fuse_sequence(raw_observations)

    # 4. Step 2: Pass through Trajectory Smoother
    smoothed_obs = smoother.smooth_trajectory(fused_obs)

    # 5. Step 3: Export final trajectory via S2Exporter
    payload = exporter.create_payload(smoothed_obs)
    out_file = tmp_path / "final_s2_output.json"
    exported_path = exporter.export_to_json(payload, out_file)

    # 6. Verify Export File Integrity
    assert exported_path.exists()
    
    with open(exported_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["coordinate_frame"] == "NED"
    assert len(data["observations"]) == 5
    assert data["observations"][0]["observation_id"] == "frame_000"
    assert data["observations"][4]["observation_id"] == "frame_004"