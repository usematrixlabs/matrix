"""
S3 Point Cloud Geometry & Filtering Utilities
"""

from typing import Optional, Tuple
import numpy as np

from ..models.s3_output import PointCloudData


class PointCloudProcessor:
    """Provides geometric filtering and post-processing for reconstructed point clouds."""

    @staticmethod
    def statistical_outlier_removal(
        point_cloud: PointCloudData,
        nb_neighbors: int = 10,
        std_ratio: float = 2.0,
    ) -> PointCloudData:
        """
        Remove statistical outliers from a point cloud based on mean neighbor distances.

        Parameters:
            point_cloud: Input PointCloudData.
            nb_neighbors: Number of nearest neighbors to compute average distance.
            std_ratio: Standard deviation multiplier threshold.

        Returns:
            Filtered PointCloudData.
        """
        pts = point_cloud.points
        n_pts = pts.shape[0]

        if n_pts <= nb_neighbors:
            return point_cloud

        # Pairwise distance matrix (for reasonably sized point clouds)
        # Compute distances to all other points
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=-1)

        # Sort along columns to get k-nearest neighbors (excluding self at index 0)
        sorted_dists = np.sort(dists, axis=1)
        mean_dists = np.mean(sorted_dists[:, 1 : nb_neighbors + 1], axis=1)

        global_mean = np.mean(mean_dists)
        global_std = np.std(mean_dists)
        threshold = global_mean + std_ratio * global_std

        inlier_mask = mean_dists <= threshold

        filtered_pts = pts[inlier_mask]
        filtered_colors = point_cloud.colors[inlier_mask] if point_cloud.colors is not None else None
        filtered_normals = point_cloud.normals[inlier_mask] if point_cloud.normals is not None else None
        filtered_conf = point_cloud.confidences[inlier_mask] if point_cloud.confidences is not None else None

        return PointCloudData(
            points=filtered_pts,
            colors=filtered_colors,
            normals=filtered_normals,
            confidences=filtered_conf,
        )

    @staticmethod
    def compute_density(point_cloud: PointCloudData) -> float:
        """
        Compute point cloud density (points per cubic meter in bounding box).

        Returns:
            Density in pts/m^3.
        """
        if point_cloud.num_points == 0:
            return 0.0

        bbox = point_cloud.bounding_box
        dx = max(bbox.max_point[0] - bbox.min_point[0], 0.01)
        dy = max(bbox.max_point[1] - bbox.min_point[1], 0.01)
        dz = max(bbox.max_point[2] - bbox.min_point[2], 0.01)
        volume = dx * dy * dz

        return float(point_cloud.num_points / volume)

