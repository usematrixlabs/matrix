"""S3 Observation Undistortion.

Applies OpenCV's established distortion model to the raw pixel
coordinates detected in S1/S3 frame images. This is the boundary that
turns the observed (u_distorted, v_distorted) pixel coordinates into
the (u_undistorted, v_undistorted) coordinates that the pinhole
projection matrix ``P = K [R | t]`` expects.

Mathematical convention
-----------------------
We use ``cv2.undistortPoints(pts, K, D, P=K, criteria=...)`` because it
is the OpenCV-recommended fast path and exactly inverts the
``cv2.projectPoints`` forward model used by the calibration procedure.

Critically we pass ``P=K`` (not the default ``P=None``). With
``P=None``, ``cv2.undistortPoints`` returns **normalized camera
coordinates** (a 3D ray direction in the camera frame, expressed in
the K-normalized space). With ``P=K`` it returns **undistorted pixel
coordinates** in the same K pixel space as the input. The latter is
what ``P = K [R | t]`` consumes, so this is the only correct option
for our pipeline.

Inputs and outputs are both in the standard OpenCV pinhole pixel
coordinate frame: origin top-left, x → right, y → down, in pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


from ..models.calibration import CameraCalibration, CameraCalibrationError


@dataclass(frozen=True)
class UndistortionResult:
    """Outcome of undistorting a set of 2D pixel coordinates.

    Attributes
    ----------
    raw_uv : numpy.ndarray
        The original (distorted) pixel coordinates, shape (N, 2).
    undistorted_uv : numpy.ndarray
        The undistorted pixel coordinates, shape (N, 2), in the same K
        pixel space as the raw input.
    applied : bool
        True when the supplied calibration had distortion coefficients
        and they were actually applied. False when the calibration had
        no distortion (``D`` was all-zero / empty) and undistortion
        would have been a no-op.
    calibration_camera_name : str
        Identifier of the calibration record used (for audit).
    """

    raw_uv: np.ndarray
    undistorted_uv: np.ndarray
    applied: bool
    calibration_camera_name: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_uv": np.asarray(self.raw_uv).tolist(),
            "undistorted_uv": np.asarray(self.undistorted_uv).tolist(),
            "applied": bool(self.applied),
            "calibration_camera_name": self.calibration_camera_name,
        }


class ObservationUndistorter:
    """Apply camera calibration distortion to a set of 2D pixel points."""

    @staticmethod
    def undistort_points(
        raw_uv: np.ndarray,
        calibration: CameraCalibration,
    ) -> UndistortionResult:
        """Undistort ``raw_uv`` (N, 2) using ``calibration``.

        The output pixel coordinates share the same K matrix as the
        input, so they remain compatible with ``P = K [R | t]``
        projection in S3's :class:`CameraPose.projection_matrix`.
        """
        if cv2 is None:
            raise CameraCalibrationError(
                "OpenCV (cv2) is required for undistortion. "
                "Install it via 'pip install opencv-python-headless'."
            )

        if not isinstance(calibration, CameraCalibration):
            raise CameraCalibrationError(
                "calibration must be a CameraCalibration instance."
            )
        pts = np.asarray(raw_uv, dtype=np.float64)
        if pts.ndim == 1 and pts.size == 2:
            pts = pts.reshape(1, 2)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise CameraCalibrationError(
                f"raw_uv must be shape (N, 2); got {pts.shape}."
            )
        if pts.shape[0] == 0:
            return UndistortionResult(
                raw_uv=pts.astype(np.float64, copy=True),
                undistorted_uv=pts.astype(np.float64, copy=True),
                applied=False,
                calibration_camera_name=calibration.camera_name,
            )

        K = np.asarray(calibration.camera_matrix, dtype=np.float64)
        D = np.asarray(calibration.distortion_coefficients, dtype=np.float64).reshape(-1, 1)

        # ``cv2.undistortPoints`` requires ``pts`` to be either (N, 1, 2)
        # for the array-of-points form or (N, 2). The (N, 1, 2) form is
        # the documented shape; we use it explicitly to be safe across
        # OpenCV versions.
        pts_in = pts.reshape(-1, 1, 2).astype(np.float64)

        # P = K means "project undistorted points back into the same K
        # pixel space as the input". This is what keeps the resulting
        # coordinates compatible with the existing pinhole projection
        # model in CameraPose.projection_matrix.
        undistorted = cv2.undistortPoints(
            pts_in,
            K,
            D,
            P=K,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-5),
        )
        undistorted = undistorted.reshape(-1, 2).astype(np.float64)

        # Determine whether undistortion actually moved the points.
        # All-zero D → OpenCV returns points unchanged. We treat that
        # as "no distortion applied" for metadata reporting.
        applied = bool(np.any(D != 0.0)) and bool(np.any(np.abs(undistorted - pts) > 1e-9))

        return UndistortionResult(
            raw_uv=pts.astype(np.float64, copy=True),
            undistorted_uv=undistorted,
            applied=applied,
            calibration_camera_name=calibration.camera_name,
        )


__all__ = ["ObservationUndistorter", "UndistortionResult"]
