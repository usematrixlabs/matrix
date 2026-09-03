"""Unit and end-to-end tests for S3 Reconstruction Pipeline."""

import json
from pathlib import Path
import numpy as np
import pytest

from reconstruction._internal.models.schema import S3Status
from reconstruction._internal.pipeline import S3ReconstructionPipeline
from tests.fixtures.synthetic_scene import generate_synthetic_uav_dataset


def test_pipeline_end_to_end_synthetic(tmp_path: Path):
    payload, gt_points, _ = generate_synthetic_uav_dataset(
        num_frames=6, num_points=30, noise_std_px=0.1, seed=42
    )

    out_dir = tmp_path / "s3_output"
    pipeline = S3ReconstructionPipeline()

    result = pipeline.run(
        input_data=payload,
        scene_id="test_scene_001",
        output_directory=out_dir,
    )

    assert result.status in [S3Status.SUCCESS, S3Status.WARNING]
    assert result.point_cloud.num_points >= 15
    assert (out_dir / "scene.ply").is_file()

    assert (out_dir / "metadata.json").is_file()

    # Read written metadata
    with open(out_dir / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["scene_id"] == "test_scene_001"
    assert meta["geometry"]["point_count"] == result.point_cloud.num_points
    assert meta["spatial_reference"]["coordinate_frame"] == "S3_LOCAL"


def test_pipeline_invalid_input(tmp_path: Path):
    pipeline = S3ReconstructionPipeline()

    # Empty payload
    invalid_data = {"observations": []}
    result = pipeline.run(input_data=invalid_data)

    assert result.status == S3Status.INVALID_INPUT
    assert result.point_cloud.num_points == 0
    assert result.failure_info is not None


def test_pipeline_raise_on_invalid():
    pipeline = S3ReconstructionPipeline()
    invalid_data = {"observations": []}

    with pytest.raises(ValueError, match="S2 Input Validation Failed"):
        pipeline.run(input_data=invalid_data, raise_on_invalid_input=True)
