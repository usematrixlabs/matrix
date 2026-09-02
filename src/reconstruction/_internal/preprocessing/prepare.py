"""
S3 Input Preprocessing & Preparation

Transforms validated S2 observations and feature tracks into normalized projection
matrices and camera ray structures for the reconstruction engine.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

from ..models.schema import FeatureObservation, S2Observation, S2Payload


@dataclass
class PreparedTrack:
    """
    Multi-view feature track prepared for 3D triangulation.

    Attributes:
        track_id: Unique track identifier.
        observation_ids: List of observation IDs viewing this track.
        points_2d: (N, 2) array of observed pixel coordinates.
        projection_matrices: List of (3, 4) camera projection matrices P = K [R | t].
        camera_centers: (N, 3) array of optical center coordinates in world frame.
        colors: Optional (N, 3) array of observed RGB colors.
    """
    track_id: str
    observation_ids: List[str]
    points_2d: np.ndarray
    projection_matrices: List[np.ndarray]
    camera_centers: np.ndarray
    colors: Optional[np.ndarray] = None


@dataclass
class PreparedReconstructionData:
    """
    Normalized data bundle prepared for the 3D reconstruction engine.
    """
    tracks: List[PreparedTrack]
    total_observations: int
    usable_observations: int
    direct_3d_points: Optional[np.ndarray] = None
    direct_3d_colors: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReconstructionDataPreparer:
    """
    Converts validated S2 payloads into engine-ready prepared data.
    """

    def __init__(self, min_views_per_track: int = 2, max_reprojection_threshold: float = 50.0) -> None:
        """
        Initialize data preparer.

        Parameters:
            min_views_per_track: Minimum number of views required to triangulate a track (default 2).
            max_reprojection_threshold: Gross outlier threshold for initial filtering.
        """
        self.min_views_per_track = min_views_per_track
        self.max_reprojection_threshold = max_reprojection_threshold

    def prepare(self, payload: S2Payload) -> PreparedReconstructionData:
        """
        Prepare an S2 payload for 3D reconstruction.

        Parameters:
            payload: Validated S2Payload.

        Returns:
            PreparedReconstructionData container.
        """
        obs_map: Dict[str, S2Observation] = {obs.observation_id: obs for obs in payload.observations}
        total_obs = len(payload.observations)

        # Precompute projection matrix and camera center for each observation
        proj_matrices: Dict[str, np.ndarray] = {}
        cam_centers: Dict[str, np.ndarray] = {}

        for obs in payload.observations:
            p_mat = obs.pose.projection_matrix(obs.camera)
            proj_matrices[obs.observation_id] = p_mat

            # Optical center in world coordinates: C = -R^T @ t
            r_mat = obs.pose.rotation_matrix
            t_vec = obs.pose.position_array
            c_vec = -r_mat.T @ t_vec
            cam_centers[obs.observation_id] = c_vec

        # Group 2D features into tracks
        track_dict: Dict[str, Dict[str, Any]] = {}
        for obs in payload.observations:
            for feat in obs.features:
                if not feat.track_id:
                    continue
                if feat.track_id not in track_dict:
                    track_dict[feat.track_id] = {
                        "obs_ids": [],
                        "points_2d": [],
                        "colors": [],
                    }
                track_dict[feat.track_id]["obs_ids"].append(obs.observation_id)
                track_dict[feat.track_id]["points_2d"].append(feat.xy)
                if feat.rgb is not None:
                    track_dict[feat.track_id]["colors"].append(feat.rgb)

        # Filter tracks by min_views
        prepared_tracks: List[PreparedTrack] = []
        for track_id, data in track_dict.items():
            if len(data["obs_ids"]) < self.min_views_per_track:
                continue

            pts_2d = np.asarray(data["points_2d"], dtype=np.float64)
            p_mats = [proj_matrices[oid] for oid in data["obs_ids"]]
            c_centers = np.array([cam_centers[oid] for oid in data["obs_ids"]], dtype=np.float64)
            
            colors_arr = None
            if len(data["colors"]) == len(data["obs_ids"]):
                colors_arr = np.asarray(data["colors"], dtype=np.uint8)

            prepared_tracks.append(PreparedTrack(
                track_id=track_id,
                observation_ids=data["obs_ids"],
                points_2d=pts_2d,
                projection_matrices=p_mats,
                camera_centers=c_centers,
                colors=colors_arr,
            ))

        return PreparedReconstructionData(
            tracks=prepared_tracks,
            total_observations=total_obs,
            usable_observations=len(proj_matrices),
            metadata={"coordinate_frame": payload.coordinate_frame, "units": payload.units},
        )

