"""
Primary S3 Reconstruction Engine Implementation
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .base import ReconstructionEngineBase
from .triangulation import MultiViewTriangulator
from ..preprocessing.prepare import PreparedReconstructionData


class DefaultReconstructionEngine(ReconstructionEngineBase):
    """
    Standard S3 reconstruction engine utilizing multi-view SVD triangulation.
    """

    def __init__(
        self,
        max_reprojection_error_px: float = 3.0,
        min_parallax_angle_deg: float = 1.0,
    ) -> None:
        """
        Initialize reconstruction engine.

        Parameters:
            max_reprojection_error_px: Threshold for reprojection error filtering.
            min_parallax_angle_deg: Minimum angular baseline separation threshold.
        """
        self.triangulator = MultiViewTriangulator(
            max_reprojection_error_px=max_reprojection_error_px,
            min_parallax_angle_deg=min_parallax_angle_deg,
        )

    def reconstruct(
        self,
        prepared_data: PreparedReconstructionData,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, Dict[str, Any]]:
        """
        Execute multi-view 3D point cloud reconstruction.

        Parameters:
            prepared_data: Prepared multi-view tracks and camera geometry.

        Returns:
            Tuple of:
                - points_3d: (N, 3) float64 array.
                - colors: Optional (N, 3) uint8 array.
                - reprojection_errors: (N,) float64 array.
                - engine_stats: Runtime statistics.
        """
        start_time = time.perf_counter()

        reconstructed_points: List[np.ndarray] = []
        reconstructed_colors: List[np.ndarray] = []
        reprojection_errors: List[float] = []
        successful_tracks = 0
        total_tracks = len(prepared_data.tracks)

        has_colors = False

        for track in prepared_data.tracks:
            pt_3d, mean_err = self.triangulator.triangulate_point_n_views(
                points_2d=track.points_2d,
                projection_matrices=track.projection_matrices,
                camera_centers=track.camera_centers,
            )

            if pt_3d is not None:
                reconstructed_points.append(pt_3d)
                reprojection_errors.append(mean_err)
                successful_tracks += 1

                # Average RGB color across observations for this track
                if track.colors is not None and len(track.colors) > 0:
                    mean_color = np.mean(track.colors, axis=0).astype(np.uint8)
                    reconstructed_colors.append(mean_color)
                    has_colors = True
                else:
                    reconstructed_colors.append(np.array([200, 200, 200], dtype=np.uint8))

        elapsed = time.perf_counter() - start_time

        # Handle direct 3D points if any were provided
        if prepared_data.direct_3d_points is not None and len(prepared_data.direct_3d_points) > 0:
            direct_pts = prepared_data.direct_3d_points
            reconstructed_points.extend(list(direct_pts))
            reprojection_errors.extend([0.0] * len(direct_pts))
            if prepared_data.direct_3d_colors is not None:
                reconstructed_colors.extend(list(prepared_data.direct_3d_colors))
                has_colors = True
            else:
                reconstructed_colors.extend([np.array([200, 200, 200], dtype=np.uint8)] * len(direct_pts))

        # Format arrays
        if len(reconstructed_points) > 0:
            points_arr = np.asarray(reconstructed_points, dtype=np.float64)
            errors_arr = np.asarray(reprojection_errors, dtype=np.float64)
            colors_arr = np.asarray(reconstructed_colors, dtype=np.uint8) if has_colors else None
        else:
            points_arr = np.empty((0, 3), dtype=np.float64)
            errors_arr = np.empty((0,), dtype=np.float64)
            colors_arr = None

        stats = {
            "total_tracks": total_tracks,
            "triangulated_points": len(points_arr),
            "triangulation_success_rate": (successful_tracks / total_tracks) if total_tracks > 0 else 0.0,
            "processing_time_s": elapsed,
        }

        return points_arr, colors_arr, errors_arr, stats

