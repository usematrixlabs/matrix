"""Unit tests for S3 Input Loader."""

import json
from pathlib import Path
import pytest

from src.reconstruction.input.loader import S2InputLoader
from src.reconstruction.models.schema import S2Payload
from tests.fixtures.synthetic_scene import generate_synthetic_uav_dataset


def test_load_from_dict_valid():
    payload, gt_points, _ = generate_synthetic_uav_dataset(num_frames=4, num_points=20)
    data = payload.to_dict()

    loader = S2InputLoader()
    loaded_payload = loader.load_from_dict(data)

    assert isinstance(loaded_payload, S2Payload)
    assert len(loaded_payload.observations) == 4
    assert loaded_payload.observations[0].camera.fx == 1200.0
    assert loaded_payload.observations[0].pose.orientation_format == "ROTATION_MATRIX"


def test_load_from_file_valid(tmp_path: Path):
    payload, _, _ = generate_synthetic_uav_dataset(num_frames=3, num_points=10)
    json_path = tmp_path / "s2_output.json"
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload.to_dict(), f)

    loader = S2InputLoader()
    loaded_payload = loader.load_from_file(json_path)

    assert len(loaded_payload.observations) == 3
    # Check absolute path resolution
    first_obs_img = Path(loaded_payload.observations[0].image_path)
    assert first_obs_img.is_absolute()
    assert first_obs_img.parent == tmp_path / "frames"


def test_load_missing_file(tmp_path: Path):
    loader = S2InputLoader()
    with pytest.raises(FileNotFoundError):
        loader.load_from_file(tmp_path / "non_existent.json")


def test_load_corrupt_json(tmp_path: Path):
    corrupt_file = tmp_path / "corrupt.json"
    corrupt_file.write_text("{ incomplete json ...")

    loader = S2InputLoader()
    with pytest.raises(ValueError, match="Corrupt S2 JSON"):
        loader.load_from_file(corrupt_file)


def test_load_invalid_type():
    loader = S2InputLoader()
    with pytest.raises(TypeError, match="Payload must be a dictionary"):
        loader.load_from_dict(["not", "a", "dict"])


def test_extract_feature_tracks():
    payload, _, _ = generate_synthetic_uav_dataset(num_frames=5, num_points=15)
    loader = S2InputLoader()
    tracks = loader.extract_feature_tracks(payload.observations)

    assert len(tracks) > 0
    # Every track should have observations across multiple views
    for track_id, obs_list in tracks.items():
        assert track_id.startswith("trk_")
        assert len(obs_list) >= 1
        assert "xy" in obs_list[0]
        assert "pose" in obs_list[0]
        assert "camera" in obs_list[0]

