"""
Full Integration tests between Subsystem S3 (3D Reconstruction) and Subsystem S4 (Georeferencing & Validation).
"""

import numpy as np
import pytest

from georeferencing_validation._internal.control_points import ControlPoints
from georeferencing_validation._internal.crs import CoordinateReference
from georeferencing_validation._internal.georeferencer import Georeferencer
from georeferencing_validation._internal.input import ReconstructionInput
from georeferencing_validation._internal.validator import GeoreferencingValidator
from reconstruction._internal.pipeline import S3ReconstructionPipeline
from tests.fixtures.synthetic_scene import generate_synthetic_uav_dataset


def test_s3_to_s4_seamless_pipeline_integration():
    payload, gt_points, _ = generate_synthetic_uav_dataset(
        num_frames=6, num_points=35, noise_std_px=0.05, seed=777
    )

    pipeline = S3ReconstructionPipeline()
    s3_result = pipeline.run(payload, scene_id="flight_test_scene")

    assert s3_result.point_cloud.num_points >= 15

    s4_input = s3_result.to_s4_reconstruction_input()
    assert isinstance(s4_input, ReconstructionInput)
    assert s4_input.num_points == s3_result.point_cloud.num_points
    assert s4_input.points.shape == (s3_result.point_cloud.num_points, 3)

    true_scale = 1.0
    true_rot = np.array([
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    true_trans = np.array([500000.0, 3000000.0, 100.0], dtype=np.float64)

    src_gcp = s4_input.points[:6]
    tgt_gcp = true_scale * (src_gcp @ true_rot.T) + true_trans

    control_points = ControlPoints(
        source=src_gcp,
        target=tgt_gcp,
    )

    source_crs = CoordinateReference.local(allow_local_to_world=True)
    target_crs = CoordinateReference.utm(zone=43)

    georeferencer = Georeferencer(
        reconstruction_data=s4_input,
        control_points=control_points,
        source_crs=source_crs,
        target_crs=target_crs,
    )
    geo_result = georeferencer.georeference(
        validate_accuracy=True,
        tolerance=0.5,
        horizontal_tolerance=0.3,
        vertical_tolerance=0.3,
    )

    assert geo_result.points.shape == s4_input.points_array.shape
    assert geo_result.transformation is not None
    assert geo_result.validation_result is not None
    assert geo_result.validation_result.passed is True
    assert geo_result.validation_result.rmse < 0.2
    assert geo_result.validation_result.horizontal_rmse < 0.2
    assert geo_result.validation_result.vertical_rmse < 0.2

    contract_payload = geo_result.export_contract_payload()
    assert contract_payload["geo_referenced_scene"]["point_cloud"]["num_points"] == s4_input.num_points
    assert contract_payload["validation_metrics"]["geometric_accuracy"] < 0.2

