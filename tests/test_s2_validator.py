"""Unit tests for S3 Input Validator."""

import copy
import pytest

from reconstruction._internal.input.validator import S2InputValidator
from reconstruction._internal.models.schema import CameraIntrinsics, CameraPose, S3Status
from tests.fixtures.synthetic_scene import generate_synthetic_uav_dataset


def test_validate_valid_payload():
    payload, _, _ = generate_synthetic_uav_dataset(num_frames=5, num_points=20)
    validator = S2InputValidator()
    report = validator.validate(payload)

    assert report.is_valid is True
    assert report.status == S3Status.SUCCESS
    assert len(report.errors) == 0
    assert report.num_valid_observations == 5


def test_validate_empty_observations():
    payload, _, _ = generate_synthetic_uav_dataset(num_frames=5, num_points=20)
    payload.observations = []
    validator = S2InputValidator()
    report = validator.validate(payload)

    assert report.is_valid is False
    assert report.status == S3Status.INVALID_INPUT
    assert "Payload contains zero observations" in report.errors[0]


def test_validate_duplicate_observation_ids():
    payload, _, _ = generate_synthetic_uav_dataset(num_frames=4, num_points=20)
    payload.observations[1].observation_id = payload.observations[0].observation_id

    validator = S2InputValidator()
    report = validator.validate(payload)

    assert report.is_valid is False
    assert any("Duplicate observation_id" in err for err in report.errors)


def test_validate_invalid_intrinsics():
    payload, _, _ = generate_synthetic_uav_dataset(num_frames=4, num_points=20)
    # Set invalid negative focal length
    payload.observations[0].camera.fx = -100.0

    validator = S2InputValidator()
    report = validator.validate(payload)

    assert report.is_valid is False
    assert any("Invalid focal length fx" in err for err in report.errors)


def test_validate_invalid_pose_position():
    payload, _, _ = generate_synthetic_uav_dataset(num_frames=4, num_points=20)
    payload.observations[0].pose.position = [float("nan"), 0.0, 10.0]

    validator = S2InputValidator()
    report = validator.validate(payload)

    assert report.is_valid is False
    assert any("Invalid pose position" in err for err in report.errors)


def test_validate_non_orthogonal_rotation():
    payload, _, _ = generate_synthetic_uav_dataset(num_frames=4, num_points=20)
    # Corrupt rotation matrix
    corrupt_rot = [
        [1.0, 2.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ]
    payload.observations[0].pose.orientation = corrupt_rot
    payload.observations[0].pose.orientation_format = "ROTATION_MATRIX"

    validator = S2InputValidator()
    report = validator.validate(payload)

    assert report.is_valid is False
    assert any("not orthogonal" in err for err in report.errors)


def test_validate_warnings_on_low_confidence():
    payload, _, _ = generate_synthetic_uav_dataset(num_frames=4, num_points=20)
    payload.observations[0].localization.confidence = 0.15

    validator = S2InputValidator()
    report = validator.validate(payload)

    assert report.is_valid is True
    assert report.status == S3Status.WARNING
    assert any("Low localization confidence" in wrn for wrn in report.warnings)


def test_raise_if_invalid():
    payload, _, _ = generate_synthetic_uav_dataset(num_frames=4, num_points=20)
    payload.observations[0].camera.fx = 0.0

    validator = S2InputValidator()
    report = validator.validate(payload)

    with pytest.raises(ValueError, match="S2 Input Validation Failed"):
        report.raise_if_invalid()

