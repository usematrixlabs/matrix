"""
S3 Input Validator

Performs strict boundary validation on S2 upstream payloads, camera calibration,
poses, coordinate consistency, and feature associations.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import numpy as np

from ..models.schema import CameraIntrinsics, CameraPose, S2Observation, S2Payload, S3Status


@dataclass
class ValidationReport:
    """
    Detailed report produced by S2 input validation.

    Attributes:
        is_valid: True if input meets all essential requirements for reconstruction.
        status: Suggested S3 status code (SUCCESS, WARNING, or INVALID_INPUT).
        errors: Fatal validation errors that prevent reconstruction.
        warnings: Non-fatal quality or consistency warnings.
        num_observations: Number of observations evaluated.
        num_valid_observations: Number of observations with valid poses and calibration.
    """
    is_valid: bool
    status: S3Status
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    num_observations: int = 0
    num_valid_observations: int = 0

    def raise_if_invalid(self) -> None:
        """Raise ValueError if the input is invalid."""
        if not self.is_valid:
            err_msg = "; ".join(self.errors)
            raise ValueError(f"S2 Input Validation Failed: {err_msg}")


class S2InputValidator:
    """
    Validates S2 payloads against the S2 -> S3 contract specifications.
    """

    def __init__(self, check_image_files: bool = False, min_observations: int = 2) -> None:
        """
        Initialize the validator.

        Parameters:
            check_image_files: If True, verifies that referenced image files exist on disk.
            min_observations: Minimum number of valid observations required (default 2).
        """
        self.check_image_files = check_image_files
        self.min_observations = min_observations

    def validate(self, payload: S2Payload) -> ValidationReport:
        """
        Validate an S2Payload object.

        Parameters:
            payload: The S2Payload instance to validate.

        Returns:
            ValidationReport detailing validation status, errors, and warnings.
        """
        errors: List[str] = []
        warnings: List[str] = []

        if not isinstance(payload, S2Payload):
            return ValidationReport(
                is_valid=False,
                status=S3Status.INVALID_INPUT,
                errors=[f"Expected S2Payload instance, got {type(payload).__name__}"],
            )

        if not payload.observations:
            return ValidationReport(
                is_valid=False,
                status=S3Status.INVALID_INPUT,
                errors=["Payload contains zero observations."],
            )

        num_obs = len(payload.observations)
        if num_obs < self.min_observations:
            errors.append(f"Insufficient observations: got {num_obs}, minimum required is {self.min_observations}")

        seen_ids = set()
        last_timestamp = -np.inf
        valid_obs_count = 0

        for i, obs in enumerate(payload.observations):
            obs_prefix = f"Observation [{i}] ({obs.observation_id}):"

            # Check unique observation IDs
            if not obs.observation_id:
                errors.append(f"{obs_prefix} Missing observation_id")
            elif obs.observation_id in seen_ids:
                errors.append(f"{obs_prefix} Duplicate observation_id '{obs.observation_id}'")
            else:
                seen_ids.add(obs.observation_id)

            # Check timestamps
            if not np.isfinite(obs.timestamp):
                errors.append(f"{obs_prefix} Non-finite timestamp: {obs.timestamp}")
            elif obs.timestamp < last_timestamp:
                warnings.append(f"{obs_prefix} Non-monotonic timestamp ({obs.timestamp} < {last_timestamp})")
            last_timestamp = obs.timestamp

            # Check image path
            if not obs.image_path:
                errors.append(f"{obs_prefix} Missing image_path")
            elif self.check_image_files:
                img_path = Path(obs.image_path)
                if not img_path.is_file():
                    errors.append(f"{obs_prefix} Image file does not exist: {img_path}")

            # Check camera calibration
            calib_errors = self._validate_intrinsics(obs.camera)
            for err in calib_errors:
                errors.append(f"{obs_prefix} {err}")

            # Check camera pose
            pose_errors, pose_warnings = self._validate_pose(obs.pose)
            for err in pose_errors:
                errors.append(f"{obs_prefix} {err}")
            for wrn in pose_warnings:
                warnings.append(f"{obs_prefix} {wrn}")

            # Check localization status
            if obs.localization.status.upper() in ["FAILED", "REJECTED"]:
                warnings.append(f"{obs_prefix} Localization status is {obs.localization.status}")
            elif obs.localization.confidence < 0.3:
                warnings.append(f"{obs_prefix} Low localization confidence ({obs.localization.confidence:.2f})")

            # Check features
            feat_errors = self._validate_features(obs)
            for err in feat_errors:
                errors.append(f"{obs_prefix} {err}")

            if not calib_errors and not pose_errors:
                valid_obs_count += 1

        is_valid = len(errors) == 0 and valid_obs_count >= self.min_observations
        status = S3Status.SUCCESS
        if not is_valid:
            status = S3Status.INVALID_INPUT
        elif len(warnings) > 0:
            status = S3Status.WARNING

        return ValidationReport(
            is_valid=is_valid,
            status=status,
            errors=errors,
            warnings=warnings,
            num_observations=num_obs,
            num_valid_observations=valid_obs_count,
        )

    @staticmethod
    def _validate_intrinsics(camera: Optional[CameraIntrinsics]) -> List[str]:
        """Validate camera intrinsic calibration parameters."""
        errors: List[str] = []
        if camera is None:
            return ["Camera intrinsics missing."]

        if not np.isfinite(camera.fx) or camera.fx <= 0:
            errors.append(f"Invalid focal length fx: {camera.fx} (must be > 0)")
        if not np.isfinite(camera.fy) or camera.fy <= 0:
            errors.append(f"Invalid focal length fy: {camera.fy} (must be > 0)")
        if not np.isfinite(camera.cx):
            errors.append(f"Non-finite principal point cx: {camera.cx}")
        if not np.isfinite(camera.cy):
            errors.append(f"Non-finite principal point cy: {camera.cy}")

        return errors

    @staticmethod
    def _validate_pose(pose: Optional[CameraPose]) -> tuple[List[str], List[str]]:
        """Validate camera pose position and orientation."""
        errors: List[str] = []
        warnings: List[str] = []

        if pose is None:
            return ["Camera pose missing."], []

        # Position check
        pos = np.asarray(pose.position, dtype=np.float64)
        if pos.shape != (3,) or not np.all(np.isfinite(pos)):
            errors.append(f"Invalid pose position: {pose.position}")

        # Orientation check
        fmt = pose.orientation_format.upper()
        if fmt == "QUATERNION_XYZW":
            q = np.asarray(pose.orientation, dtype=np.float64)
            if q.shape != (4,) or not np.all(np.isfinite(q)):
                errors.append(f"Invalid quaternion orientation: {pose.orientation}")
            else:
                norm = np.linalg.norm(q)
                if np.isclose(norm, 0.0):
                    errors.append("Zero-norm quaternion orientation")
                elif not np.isclose(norm, 1.0, atol=1e-2):
                    warnings.append(f"Unnormalized quaternion (norm = {norm:.4f})")
        elif fmt == "ROTATION_MATRIX":
            r = np.asarray(pose.orientation, dtype=np.float64)
            if r.shape != (3, 3) or not np.all(np.isfinite(r)):
                errors.append(f"Invalid rotation matrix: {pose.orientation}")
            else:
                # Check orthogonality R^T R ≈ I
                r_rt = r.T @ r
                if not np.allclose(r_rt, np.eye(3), atol=1e-3):
                    errors.append("Rotation matrix is not orthogonal (R^T R != I)")
                det = np.linalg.det(r)
                if not np.isclose(det, 1.0, atol=1e-3):
                    errors.append(f"Rotation matrix determinant != 1.0 (det = {det:.4f})")
        else:
            errors.append(f"Unknown orientation format: '{pose.orientation_format}'")

        return errors, warnings

    @staticmethod
    def _validate_features(obs: S2Observation) -> List[str]:
        """Validate 2D feature observations."""
        errors: List[str] = []
        for j, feat in enumerate(obs.features):
            if not feat.feature_id:
                errors.append(f"Feature [{j}] missing feature_id")
            xy = np.asarray(feat.xy, dtype=np.float64)
            if xy.shape != (2,) or not np.all(np.isfinite(xy)):
                errors.append(f"Feature [{j}] ({feat.feature_id}) has invalid coordinates: {feat.xy}")
            if feat.rgb is not None:
                rgb = np.asarray(feat.rgb)
                if rgb.shape != (3,) or np.any(rgb < 0) or np.any(rgb > 255):
                    errors.append(f"Feature [{j}] ({feat.feature_id}) has invalid RGB: {feat.rgb}")
        return errors

