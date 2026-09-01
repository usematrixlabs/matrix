import numpy as np
import pytest

from src.georeferencing_validation.control_points import ControlPoints
from src.georeferencing_validation.crs import CoordinateReference
from src.georeferencing_validation.georeferencer import Georeferencer
from src.georeferencing_validation.input import ReconstructionInput


def test_georeferencer_rejects_local_to_projected_without_explicit_policy():
    source = CoordinateReference(
        name="S3_LOCAL",
        epsg=None,
        units="meters",
        frame_type="local",
    )
    target = CoordinateReference(
        name="UTM Zone 43N",
        epsg=32643,
        units="meters",
        frame_type="projected",
        projection="UTM",
        datum="WGS84",
    )

    reconstruction = ReconstructionInput(
        points=np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
    )

    control = ControlPoints(
        source=np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64),
        target=np.array([
            [500000.0, 3000000.0, 10.0],
            [500001.0, 3000000.0, 10.0],
            [500000.0, 3000001.0, 10.0],
            [500000.0, 3000000.0, 11.0],
        ], dtype=np.float64),
    )

    with pytest.raises(ValueError, match="local-to-world"):
        Georeferencer(
            reconstruction_data=reconstruction,
            control_points=control,
            source_crs=source,
            target_crs=target,
        )


def test_georeferencer_accepts_explicit_local_to_world_policy():
    source = CoordinateReference(
        name="S3_LOCAL",
        epsg=None,
        units="meters",
        frame_type="local",
        allow_local_to_world=True,
    )
    target = CoordinateReference(
        name="UTM Zone 43N",
        epsg=32643,
        units="meters",
        frame_type="projected",
        projection="UTM",
        datum="WGS84",
    )

    reconstruction = ReconstructionInput(
        points=np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
    )

    control = ControlPoints(
        source=np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64),
        target=np.array([
            [500000.0, 3000000.0, 100.0],
            [500001.0, 3000000.0, 100.0],
            [500000.0, 3000001.0, 100.0],
            [500000.0, 3000000.0, 101.0],
        ], dtype=np.float64),
    )

    georeferencer = Georeferencer(
        reconstruction_data=reconstruction,
        control_points=control,
        source_crs=source,
        target_crs=target,
    )
    result = georeferencer.georeference()

    assert result.metadata["coordinate_frame_mode"] == "explicit_local_to_world"
    assert result.points.shape == reconstruction.points_array.shape
