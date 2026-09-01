import json
from pathlib import Path

from src.localization_sensor_fusion.exporters.s2_exporter import S2Exporter
from src.localization_sensor_fusion.schemas.contracts import (
    CameraPose,
    LocalizationMeta,
    LocalizationQuality,
    LocalizationSource,
    Position,
    QuaternionOrientation,
    S2ObservationOutput,
)


def test_s2_exporter_json_creation(tmp_path: Path):
    exporter = S2Exporter(coordinate_frame="NED", units="meters")

    dummy_obs = S2ObservationOutput(
        observation_id="frame_001",
        timestamp=100.0,
        image="frame_001.jpg",
        pose=CameraPose(
            position=Position(x=1.0, y=2.0, z=3.0),
            orientation=QuaternionOrientation(qw=1.0, qx=0.0, qy=0.0, qz=0.0),
        ),
        localization=LocalizationMeta(
            status="estimated",
            source=[LocalizationSource.VISUAL],
            quality=LocalizationQuality(confidence=0.95),
        ),
    )

    payload = exporter.create_payload([dummy_obs])
    out_file = tmp_path / "output_payload.json"
    exported_path = exporter.export_to_json(payload, out_file)

    assert exported_path.exists()

    with open(exported_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["coordinate_frame"] == "NED"
    assert data["units"]["position"] == "meters"
    assert len(data["observations"]) == 1
    assert data["observations"][0]["observation_id"] == "frame_001"
    assert data["observations"][0]["localization"]["status"] == "estimated"