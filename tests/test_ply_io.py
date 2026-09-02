"""Unit tests for PLY file input/output."""

from pathlib import Path
import numpy as np
import pytest

from src.reconstruction._internal.geometry.ply_io import PlyIO
from src.reconstruction._internal.models.s3_output import PointCloudData


def test_ply_binary_roundtrip(tmp_path: Path):
    points = np.array([
        [1.0, 2.0, 3.0],
        [4.5, -2.1, 0.8],
        [10.0, 20.0, 30.0],
    ], dtype=np.float64)
    colors = np.array([
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
    ], dtype=np.uint8)

    cloud = PointCloudData(points=points, colors=colors)
    ply_path = tmp_path / "test_binary.ply"

    PlyIO.write_ply(ply_path, cloud, binary=True)
    assert ply_path.is_file()

    loaded_cloud = PlyIO.read_ply(ply_path)
    assert loaded_cloud.num_points == 3
    np.testing.assert_allclose(loaded_cloud.points, points, atol=1e-5)
    np.testing.assert_array_equal(loaded_cloud.colors, colors)


def test_ply_ascii_roundtrip(tmp_path: Path):
    points = np.array([
        [-10.2, 5.4, 1.2],
        [3.3, 0.0, -8.1],
    ], dtype=np.float64)
    colors = np.array([
        [128, 128, 128],
        [200, 100, 50],
    ], dtype=np.uint8)

    cloud = PointCloudData(points=points, colors=colors)
    ply_path = tmp_path / "test_ascii.ply"

    PlyIO.write_ply(ply_path, cloud, binary=False)
    assert ply_path.is_file()

    loaded_cloud = PlyIO.read_ply(ply_path)
    assert loaded_cloud.num_points == 2
    np.testing.assert_allclose(loaded_cloud.points, points, atol=1e-5)
    np.testing.assert_array_equal(loaded_cloud.colors, colors)


def test_empty_ply_io(tmp_path: Path):
    cloud = PointCloudData(points=np.empty((0, 3), dtype=np.float64))
    ply_path = tmp_path / "empty.ply"

    PlyIO.write_ply(ply_path, cloud, binary=True)
    loaded = PlyIO.read_ply(ply_path)
    assert loaded.num_points == 0

