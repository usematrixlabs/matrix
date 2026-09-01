"""Visual Localization Engine utilizing OpenCV PnP and Feature Matching."""

from typing import Optional, Tuple

import numpy as np

try:
    import cv2
except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional runtime package
    raise ModuleNotFoundError(
        "OpenCV is required for the visual localization engine. "
        "Install it with: pip install opencv-python-headless"
    ) from exc


class VisualLocalizerEngine:
    """Estimates 3D camera pose (R, t) from 2D-3D point correspondences using PnP."""

    def __init__(self, camera_matrix: np.ndarray, dist_coeffs: Optional[np.ndarray] = None):
        self.K = camera_matrix
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros((4, 1))

    def estimate_pose_pnp(
        self, object_points: np.ndarray, image_points: np.ndarray
    ) -> Tuple[bool, np.ndarray, np.ndarray]:
        """Solves Perspective-n-Point problem using RANSAC.
        
        Args:
            object_points: (N, 3) matrix of 3D world coordinates.
            image_points: (N, 2) matrix of 2D image coordinates.
        """
        if len(object_points) < 4 or len(image_points) < 4:
            return False, np.zeros((3, 1)), np.zeros((3, 1))

        success, rvec, tvec, _ = cv2.solvePnPRansac(
            object_points.astype(np.float32),
            image_points.astype(np.float32),
            self.K.astype(np.float32),
            self.dist_coeffs.astype(np.float32),
        )

        return success, rvec, tvec