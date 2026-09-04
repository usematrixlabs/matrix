"""
S3 Input Preprocessing & Preparation

Transforms validated S2 observations and feature tracks into normalized projection
matrices and camera ray structures for the reconstruction engine.

Distortion handling
-------------------
When a :class:`CameraCalibration` is supplied to :class:`ReconstructionDataPreparer`,
each track's observed pixel coordinates are undistorted via
:class:`ObservationUndistorter` before triangulation. The raw and undistorted
coordinates are both preserved on the :class:`PreparedTrack` so that:

* downstream code can still report what the detector actually saw,
* ``metadata.camera_calibration`` can report whether undistortion was applied,
* the projection model ``P = K [R | t]`` operates on undistorted pixel
  coordinates (consistent with what :func:`cv2.undistortPoints` returns when
  ``P = K``).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

from ..models.schema import FeatureObservation, S2Observation, S2Payload
from ..models.calibration import CameraCalibration, CameraCalibrationError
from .undistort import ObservationUndistorter


@dataclass
class PreparedTrack:
    """
    Multi-view feature track prepared for 3D triangulation.

    Attributes:
        track_id: Unique track identifier.
        observation_ids: List of observation IDs viewing this track.
        points_2d: (N, 2) array of pixel coordinates consumed by triangulation.
            These are the **undistorted** pixel coordinates when a calibration
            was supplied to the preparer; otherwise they equal ``points_2d_raw``.
        points_2d_raw: (N, 2) array of observed (distorted) pixel coordinates
            as produced by the upstream detector. Always populated; useful
            for diagnostics and reprojection-error cross-checks against the
            original observation.
        projection_matrices: List of (3, 4) camera projection matrices P = K [R | t].
        camera_centers: (N, 3) array of optical center coordinates in world frame.
        colors: Optional (N, 3) array of observed RGB colors.
        undistortion_applied: True if the preparer ran ``cv2.undistortPoints``
            on this track's observations using the supplied calibration.
    """
    track_id: str
    observation_ids: List[str]
    points_2d: np.ndarray
    projection_matrices: List[np.ndarray]
    camera_centers: np.ndarray
    colors: Optional[np.ndarray] = None
    points_2d_raw: Optional[np.ndarray] = None
    undistortion_applied: bool = False


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

    def __init__(
        self,
        min_views_per_track: int = 2,
        max_reprojection_threshold: float = 50.0,
        calibration: Optional[CameraCalibration] = None,
    ) -> None:
        """
        Initialize data preparer.

        Parameters:
            min_views_per_track: Minimum number of views required to triangulate a track (default 2).
            max_reprojection_threshold: Gross outlier threshold for initial filtering.
            calibration: Optional :class:`CameraCalibration` to apply distortion
                correction to. When supplied, all observation pixel coordinates
                are undistorted before being placed on the prepared tracks. The
                calibration must be pre-scaled to the actual video resolution;
                use :meth:`CameraCalibration.scale_to_resolution` if needed.
        """
        self.min_views_per_track = min_views_per_track
        self.max_reprojection_threshold = max_reprojection_threshold
        self.calibration = calibration

    def set_calibration(self, calibration: Optional[CameraCalibration]) -> None:
        """Attach (or clear) the calibration record used for undistortion."""
        self.calibration = calibration

    def prepare(self, payload: S2Payload) -> PreparedReconstructionData:
        """
        Prepare an S2 payload for 3D reconstruction.

        Parameters:
            payload: Validated S2Payload.

        Returns:
            PreparedReconstructionData container. When a calibration was
            supplied, each :class:`PreparedTrack` carries both
            ``points_2d_raw`` (distorted detector output) and
            ``points_2d`` (undistorted), and the container's metadata
            records ``"camera_calibration"`` with the calibration summary
            and ``"undistortion_applied"`` boolean.
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

        # Apply calibration-based undistortion when available
        undistortion_summary: Dict[str, Any] = {
            "enabled": self.calibration is not None,
            "applied": False,
        }
        if self.calibration is not None:
            undistortion_summary["camera_name"] = self.calibration.camera_name
            undistortion_summary["image_width"] = self.calibration.image_width
            undistortion_summary["image_height"] = self.calibration.image_height
            undistortion_summary["distortion_model"] = self.calibration.distortion_model
            undistortion_summary["distortion_coefficients"] = (
                self.calibration.distortion_coefficients.tolist()
            )

        # Filter tracks by min_views
        prepared_tracks: List[PreparedTrack] = []
        any_track_undistorted = False
        for track_id, data in track_dict.items():
            if len(data["obs_ids"]) < self.min_views_per_track:
                continue

            raw_pts_2d = np.asarray(data["points_2d"], dtype=np.float64)
            p_mats = [proj_matrices[oid] for oid in data["obs_ids"]]
            c_centers = np.array([cam_centers[oid] for oid in data["obs_ids"]], dtype=np.float64)

            colors_arr = None
            if len(data["colors"]) == len(data["obs_ids"]):
                colors_arr = np.asarray(data["colors"], dtype=np.uint8)

            track_undistortion_applied = False
            if self.calibration is not None:
                try:
                    und = ObservationUndistorter.undistort_points(
                        raw_pts_2d, self.calibration
                    )
                    triangulation_pts = und.undistorted_uv
                    track_undistortion_applied = bool(und.applied)
                    if track_undistortion_applied:
                        any_track_undistorted = True
                except CameraCalibrationError:
                    triangulation_pts = raw_pts_2d
            else:
                triangulation_pts = raw_pts_2d

            prepared_tracks.append(PreparedTrack(
                track_id=track_id,
                observation_ids=data["obs_ids"],
                points_2d=triangulation_pts,
                points_2d_raw=raw_pts_2d.copy(),
                projection_matrices=p_mats,
                camera_centers=c_centers,
                colors=colors_arr,
                undistortion_applied=track_undistortion_applied,
            ))

        undistortion_summary["applied"] = any_track_undistorted

        meta: Dict[str, Any] = {
            "coordinate_frame": payload.coordinate_frame,
            "units": payload.units,
        }
        if self.calibration is not None:
            meta["camera_calibration"] = self.calibration.to_dict()
            meta["undistortion_applied"] = bool(any_track_undistorted)

        return PreparedReconstructionData(
            tracks=prepared_tracks,
            total_observations=total_obs,
            usable_observations=len(proj_matrices),
            metadata=meta,
        )

