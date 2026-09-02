"""Visual Localizer Engine for 2D-to-3D PnP Pose Estimation and Feature Matching."""

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from ..schemas.contracts import (
    CameraPose,
    Position,
    QuaternionOrientation,
    LocalizationQuality,
)


class VisualLocalizerEngine:
    """
    Performs 2D-to-3D local pose estimation using OpenCV PnP RANSAC solvers.
    Includes built-in ORB feature detection & matching for raw image frames.

    Note on Scope: This engine is strictly responsible for single-frame local
    pose estimation. Multi-view global optimization (Bundle Adjustment) is
    outside the scope of Subsystem 2 and is delegated to downstream 3D
    reconstruction pipelines (e.g., Subsystem 3/4).
    """

    def __init__(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray = None):
        if cv2 is None:
            raise ImportError(
                "OpenCV (cv2) is required for VisualLocalizerEngine. "
                "Install it via 'pip install opencv-python-headless'."
            )
        self.camera_matrix = np.array(camera_matrix, dtype=np.float64)
        self.dist_coeffs = (
            np.array(dist_coeffs, dtype=np.float64)
            if dist_coeffs is not None
            else np.zeros((4, 1), dtype=np.float64)
        )
        # Initialize ORB detector and Hamming Distance Matcher
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def extract_and_match_features(
        self, 
        frame: np.ndarray, 
        map_descriptors: np.ndarray, 
        map_3d_points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extracts ORB features from a raw image frame and matches them with 3D map descriptors.

        :param frame: Grayscale or BGR image frame (np.ndarray)
        :param map_descriptors: Descriptors corresponding to reference 3D points
        :param map_3d_points: Reference 3D world points (Nx3)
        :return: Tuple of matched (2D image points, 3D map points)
        """
        if frame is None or map_descriptors is None or len(map_descriptors) == 0:
            return np.empty((0, 2)), np.empty((0, 3))

        # Convert to grayscale if BGR
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        if descriptors is None or len(descriptors) == 0:
            return np.empty((0, 2)), np.empty((0, 3))

        # Match descriptors using Hamming distance
        matches = self.matcher.match(descriptors, map_descriptors)
        if not matches:
            return np.empty((0, 2)), np.empty((0, 3))

        image_pts = np.float32([keypoints[m.queryIdx].pt for m in matches])
        object_pts = np.float32([map_3d_points[m.trainIdx] for m in matches])

        return image_pts, object_pts

    def estimate_pose(
        self, image_points: np.ndarray, object_points: np.ndarray
    ) -> tuple[CameraPose | None, LocalizationQuality]:
        """
        Estimates 3D camera pose from 2D-3D point correspondences using PnP RANSAC.

        :param image_points: Nx2 numpy array of 2D pixel coordinates
        :param object_points: Nx3 numpy array of 3D world coordinates
        :return: Tuple of (CameraPose, LocalizationQuality)
        """
        if len(image_points) < 4 or len(object_points) < 4:
            return None, LocalizationQuality(confidence=0.0)

        image_pts = np.ascontiguousarray(image_points, dtype=np.float64)
        object_pts = np.ascontiguousarray(object_points, dtype=np.float64)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_pts,
            image_pts,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success or inliers is None:
            return None, LocalizationQuality(confidence=0.0)

        # Convert rotation vector to 3x3 rotation matrix using Rodrigues
        R, _ = cv2.Rodrigues(rvec)

        # Calculate unit quaternion from rotation matrix
        qw, qx, qy, qz = self._rot_matrix_to_quaternion(R)

        # Calculate accurate camera center in world coordinates: C = -R.T * tvec
        camera_position_world = -R.T @ tvec.reshape(3, 1)
        tx, ty, tz = camera_position_world.flatten()

        pose = CameraPose(
            position=Position(x=float(tx), y=float(ty), z=float(tz)),
            orientation=QuaternionOrientation(qw=qw, qx=qx, qy=qy, qz=qz),
        )

        # Confidence calculated as inlier ratio
        confidence = float(len(inliers) / len(image_points))
        quality = LocalizationQuality(confidence=min(max(confidence, 0.0), 1.0))

        return pose, quality

    def estimate_pose_from_frame(
        self, 
        frame: np.ndarray, 
        map_descriptors: np.ndarray, 
        map_3d_points: np.ndarray
    ) -> tuple[CameraPose | None, LocalizationQuality]:
        """Convenience method to extract ORB features from raw image and solve PnP in one call."""
        image_pts, object_pts = self.extract_and_match_features(frame, map_descriptors, map_3d_points)
        return self.estimate_pose(image_pts, object_pts)

    @staticmethod
    def _rot_matrix_to_quaternion(R: np.ndarray) -> tuple[float, float, float, float]:
        """Converts a 3x3 rotation matrix to normalized quaternion (qw, qx, qy, qz)."""
        tr = np.trace(R)
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S

        norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
        return float(qw / norm), float(qx / norm), float(qy / norm), float(qz / norm)

# Alias for legacy references expecting the Colmap naming convention
ColmapLocalizationEngine = VisualLocalizerEngine