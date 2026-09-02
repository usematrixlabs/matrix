"""
N-View Direct Linear Transformation (DLT) Triangulation Engine

Implements robust SVD-based multi-view triangulation with positive depth (cheirality)
checks, parallax angle verification, and reprojection error calculations.
"""

from typing import List, Optional, Tuple
import numpy as np


class MultiViewTriangulator:
    """
    Solves 3D landmark positions from N (>=2) calibrated camera views using SVD DLT.
    """

    def __init__(
        self,
        max_reprojection_error_px: float = 3.0,
        min_parallax_angle_deg: float = 1.0,
        min_positive_depth: float = 0.05,
    ) -> None:
        """
        Initialize the triangulator.

        Parameters:
            max_reprojection_error_px: Maximum acceptable mean reprojection error in pixels.
            min_parallax_angle_deg: Minimum angular separation between observing camera rays.
            min_positive_depth: Minimum forward optical depth (Z > 0) in all cameras.
        """
        self.max_reprojection_error_px = float(max_reprojection_error_px)
        self.min_parallax_angle_deg = float(min_parallax_angle_deg)
        self.min_positive_depth = float(min_positive_depth)

    def triangulate_point_n_views(
        self,
        points_2d: np.ndarray,
        projection_matrices: List[np.ndarray],
        camera_centers: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Triangulate a single 3D point from N camera views.

        Parameters:
            points_2d: (N, 2) array of observed pixel coordinates (u, v).
            projection_matrices: List of N (3, 4) projection matrices P_i = K_i [R_i | t_i].
            camera_centers: Optional (N, 3) optical center coordinates for parallax verification.

        Returns:
            Tuple of (point_3d, mean_reprojection_error).
            Returns (None, inf) if triangulation is invalid, ill-conditioned, or behind cameras.
        """
        n_views = len(points_2d)
        if n_views < 2 or len(projection_matrices) != n_views:
            return None, float("inf")

        # Construct 2N x 4 linear system A
        # For each view i:
        # A[2i]   = u_i * P_i[2] - P_i[0]
        # A[2i+1] = v_i * P_i[2] - P_i[1]
        a_mat = np.zeros((2 * n_views, 4), dtype=np.float64)

        for i in range(n_views):
            u, v = points_2d[i]
            p = projection_matrices[i]
            a_mat[2 * i] = u * p[2] - p[0]
            a_mat[2 * i + 1] = v * p[2] - p[1]

        # Solve A X = 0 using SVD
        try:
            _, _, vh = np.linalg.svd(a_mat)
        except np.linalg.LinAlgError:
            return None, float("inf")

        point_4d = vh[-1]

        # Check for point at infinity
        if np.isclose(point_4d[3], 0.0, atol=1e-12):
            return None, float("inf")

        point_3d = point_4d[:3] / point_4d[3]

        if not np.all(np.isfinite(point_3d)):
            return None, float("inf")

        point_homog = np.append(point_3d, 1.0)

        # 1. Cheirality check: Ensure point is in front of all observing cameras
        errors: List[float] = []
        for i in range(n_views):
            p = projection_matrices[i]
            proj_h = p @ point_homog
            depth = proj_h[2]

            if depth <= self.min_positive_depth:
                return None, float("inf")

            u_proj = proj_h[0] / depth
            v_proj = proj_h[1] / depth

            u_obs, v_obs = points_2d[i]
            err = float(np.hypot(u_proj - u_obs, v_proj - v_obs))
            errors.append(err)

        mean_error = float(np.mean(errors))

        # 2. Max Reprojection Error Threshold check
        if mean_error > self.max_reprojection_error_px:
            return None, mean_error

        # 3. Parallax Angle Check (if camera centers provided)
        if camera_centers is not None and len(camera_centers) >= 2:
            max_parallax_deg = self._compute_max_parallax(point_3d, camera_centers)
            if max_parallax_deg < self.min_parallax_angle_deg:
                return None, mean_error

        return point_3d, mean_error

    @staticmethod
    def _compute_max_parallax(point_3d: np.ndarray, camera_centers: np.ndarray) -> float:
        """Compute the maximum angular separation in degrees between any pair of optical rays."""
        # Vectors from camera centers to the 3D point
        rays = point_3d - camera_centers
        norms = np.linalg.norm(rays, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        unit_rays = rays / norms

        # Cosine of angles between all pairs of rays: dot products
        cosine_matrix = unit_rays @ unit_rays.T
        cosine_matrix = np.clip(cosine_matrix, -1.0, 1.0)
        
        # Minimum cosine corresponds to maximum angle
        min_cos = float(np.min(cosine_matrix))
        max_angle_rad = float(np.arccos(min_cos))
        return float(np.degrees(max_angle_rad))

