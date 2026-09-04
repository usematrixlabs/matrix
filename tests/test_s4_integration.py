"""
Integration tests between Subsystem S3 (3D Reconstruction) and Subsystem S4 (Georeferencing & Validation).
"""

import numpy as np
import pytest

from georeferencing_validation._internal.control_points import ControlPoints
from georeferencing_validation._internal.crs import CoordinateReference
from georeferencing_validation._internal.georeferencer import Georeferencer
from georeferencing_validation._internal.input import ReconstructionInput
from georeferencing_validation._internal.validator import GeoreferencingValidator
from reconstruction._internal.pipeline import S3ReconstructionPipeline
from fixtures.synthetic_scene import generate_synthetic_uav_dataset


def test_s3_to_s4_seamless_integration():
    # 1. Generate synthetic UAV flight data
    payload, gt_points, _ = generate_synthetic_uav_dataset(
        num_frames=6, num_points=35, noise_std_px=0.05, seed=777
    )

    # 2. Run S3 reconstruction pipeline
    pipeline = S3ReconstructionPipeline()
    s3_result = pipeline.run(payload, scene_id="flight_test_scene")

    assert s3_result.point_cloud.num_points >= 15

    # 3. Build S4 ReconstructionInput from S3 point cloud (no cross-subsystem import)
    s4_input = ReconstructionInput(
        points=s3_result.point_cloud.points,
        colors=s3_result.point_cloud.colors,
        metadata=s3_result.metadata if hasattr(s3_result, "metadata") else {},
    )
    assert isinstance(s4_input, ReconstructionInput)
    assert s4_input.num_points == s3_result.point_cloud.num_points
    assert s4_input.points.shape == (s3_result.point_cloud.num_points, 3)

    # 4. Set up synthetic S4 Ground Control Points (GCPs)
    # Define known source points in S3 local frame and target points in georeferenced metric frame (e.g. UTM)
    # Let target = scale * R @ source + T
    true_scale = 1.0
    true_rot = np.array([
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    true_trans = np.array([500000.0, 3000000.0, 100.0], dtype=np.float64)

    # Pick 4 reconstructed points as control points
    src_gcp = s4_input.points[:4]

    tgt_gcp = true_scale * (src_gcp @ true_rot.T) + true_trans

    control_points = ControlPoints(
        source=src_gcp,
        target=tgt_gcp,
    )

    source_crs = CoordinateReference(
        name="S3_LOCAL",
        epsg=None,
        units="meters",
    )
    target_crs = CoordinateReference(
        name="UTM Zone 43N",
        epsg=32643,
        units="meters",
    )


    # 5. Execute S4 Georeferencing Pipeline
    georeferencer = Georeferencer(
        reconstruction_data=s4_input,
        control_points=control_points,
        source_crs=source_crs,
        target_crs=target_crs,
    )
    geo_result = georeferencer.georeference()

    assert geo_result.points.shape == s4_input.points_array.shape
    assert geo_result.transformation is not None

    # 6. Validate accuracy with S4 Validator
    validator = GeoreferencingValidator(
        control_points=control_points,
        transformation=georeferencer.transformation,
        tolerance=0.5,
    )
    val_result = validator.validate()

    assert val_result.passed is True
    assert val_result.rmse < 0.2
