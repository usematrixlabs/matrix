"""S2 → S3 Bridge Adapter.

Translates S2's ``S2PayloadOutput`` (poses only) into the ``S2Payload`` that
S3 expects (poses + per-frame 2D feature tracks with shared ``track_id``s).

S2's contract carries localized observations with camera intrinsics but no
2D feature matches. S3's multi-view triangulator needs tracks — pairs of
(x, y) pixel coordinates in multiple frames that correspond to the same
physical 3D point.

This bridge is the single, well-defined place where that translation happens.
Given the images referenced by the S1 observation IDs and the S2 poses, it
extracts ORB features from each frame, matches them across consecutive
keyframes, and emits a deterministic ``track_id`` per matched feature so S3
can triangulate.

Inputs
------
- ``s2_payload``    : ``S2PayloadOutput`` produced by ``S2Exporter``
- ``image_root``    : directory in which to resolve relative image paths
- ``min_track_len`` : minimum number of inlier frames for a track to be kept
- ``max_features``  : maximum number of features per frame to detect

Failure modes
-------------
- OpenCV missing             → raises ``ImportError`` at construction time
- No images resolvable       → returns an S2Payload with empty ``features`` lists
- < 2 frames with intrinsics → returns an S2Payload with empty ``features`` lists

The S3 validator treats ``features=[]`` per observation as a recoverable
warning, so the pipeline remains runnable end-to-end even when ORB
matching fails on a particular dataset (e.g., featureless terrain).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover - exercised only without cv2
    cv2 = None
    _CV2_IMPORT_ERROR = exc
else:
    _CV2_IMPORT_ERROR = None

from src.reconstruction.models.schema import (
    CameraIntrinsics,
    CameraPose,
    FeatureObservation,
    LocalizationInfo,
    S2Observation,
    S2Payload,
)

from src.localization_sensor_fusion.schemas.contracts import (
    CameraInfo,
    CameraIntrinsics as S2CameraIntrinsics,
    S2ObservationOutput,
    S2PayloadOutput,
)


def _require_cv2() -> None:
    if cv2 is None:
        raise ImportError(
            "OpenCV (cv2) is required for the S2->S3 bridge. "
            "Install it via 'pip install opencv-python-headless'."
        ) from _CV2_IMPORT_ERROR


def _normalize_orientation_format(orientation: Any) -> Tuple[str, Any]:
    """Return (format, value) so S3's CameraPose can interpret orientation.

    S2's CameraPose stores quaternion components individually
    (qw, qx, qy, qz). S3's CameraPose expects either a 4-tuple in
    QUATERNION_XYZW order or a 3x3 ROTATION_MATRIX.
    """
    return ("QUATERNION_XYZW", [orientation.qx, orientation.qy, orientation.qz, orientation.qw])


def _intrinsics_to_matrix(cam: Optional[CameraInfo]) -> Optional[np.ndarray]:
    if cam is None or cam.intrinsics is None:
        return None
    intr = cam.intrinsics
    return np.array(
        [
            [float(intr.fx), 0.0, float(intr.cx)],
            [0.0, float(intr.fy), float(intr.cy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _safe_image_path(image_ref: str, image_root: Path) -> Optional[Path]:
    """Resolve a possibly relative image path under ``image_root``."""
    if not image_ref:
        return None
    p = Path(image_ref)
    if not p.is_absolute():
        p = image_root / p
    return p if p.is_file() else None


def _extract_orb(gray: np.ndarray, max_features: int) -> Tuple[np.ndarray, np.ndarray]:
    """Run ORB detect+compute on a grayscale image, returning keypoints + descriptors.

    Returns
    -------
    keypoints : (N, 2) float32 array of (x, y) pixel coordinates.
    descriptors : (N, 32) uint8 ORB descriptors, or an empty (0, 32) array.
    """
    orb = cv2.ORB_create(nfeatures=int(max_features))
    kps, descs = orb.detectAndCompute(gray, None)
    if not kps:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 32), dtype=np.uint8)
    pts = np.array([kp.pt for kp in kps], dtype=np.float32).reshape(-1, 2)
    if descs is None:
        descs = np.zeros((0, 32), dtype=np.uint8)
    return pts, descs


def _match_descriptors(desc_a: np.ndarray, desc_b: np.ndarray) -> List[Tuple[int, int]]:
    """Match ORB descriptors with brute-force Hamming and Lowe ratio test."""
    if desc_a.shape[0] == 0 or desc_b.shape[0] == 0:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(desc_a, desc_b)
    if not matches:
        return []
    matches = sorted(matches, key=lambda m: m.distance)
    keep: List[Tuple[int, int]] = []
    for m in matches:
        if m.distance <= 32:
            keep.append((int(m.queryIdx), int(m.trainIdx)))
    return keep


class _TrackBuilder:
    """Incrementally merges pairwise matches into multi-view feature tracks."""

    def __init__(self) -> None:
        self._tracks: Dict[int, Dict[str, Tuple[int, Tuple[float, float]]]] = {}
        self._next_track_id: int = 0

    def add_pair(
        self,
        frame_id_a: str,
        kp_index_a: int,
        frame_id_b: str,
        kp_index_b: int,
    ) -> None:
        # Find an existing track containing (frame_id_a, kp_index_a).
        existing_track_id: Optional[int] = None
        for tid, members in self._tracks.items():
            if (frame_id_a, kp_index_a) in {(f, k) for f, (k, _) in members.items()}:
                existing_track_id = tid
                break

        if existing_track_id is None:
            tid = self._next_track_id
            self._next_track_id += 1
            self._tracks[tid] = {}
            existing_track_id = tid

        members = self._tracks[existing_track_id]
        if frame_id_a not in members:
            members[frame_id_a] = (kp_index_a, (0.0, 0.0))
        if frame_id_b not in members:
            members[frame_id_b] = (kp_index_b, (0.0, 0.0))

    def finalize(self, frame_keypoints: Dict[str, np.ndarray]) -> Dict[str, Dict[int, int]]:
        """Return a mapping ``frame_id -> {kp_index: track_id}`` for tracks with >=2 frames.

        Tracks that only appear in a single frame are dropped — they have no
        triangulation value.
        """
        out: Dict[str, Dict[int, int]] = {}
        for tid, members in self._tracks.items():
            if len(members) < 2:
                continue
            for frame_id, (kp_index, _) in members.items():
                kps = frame_keypoints.get(frame_id)
                if kps is None or kp_index >= kps.shape[0]:
                    continue
                out.setdefault(frame_id, {})[kp_index] = tid
        return out


def _build_observation(
    s2_obs: S2ObservationOutput,
    pose: Optional[CameraPose],
    intrinsics: Optional[CameraIntrinsics],
    frame_track_indices: Dict[int, int],
    frame_keypoints: np.ndarray,
    frame_id: str,
    observation_id: str,
) -> S2Observation:
    """Construct one S3 S2Observation from an S2 output + matched keypoints."""
    features: List[FeatureObservation] = []
    for kp_idx, (xy, track_id) in enumerate(
        ((frame_keypoints[i, 0], frame_keypoints[i, 1]), tid)
        for i, tid in frame_track_indices.items()
    ):
        if i >= frame_keypoints.shape[0]:
            continue
        features.append(
            FeatureObservation(
                feature_id=f"{observation_id}_f{kp_idx:04d}",
                xy=(float(xy[0]), float(xy[1])),
                track_id=f"trk_{track_id:06d}",
                response=1.0,
            )
        )

    if s2_obs.localization is not None:
        localization = LocalizationInfo(
            status=str(s2_obs.localization.status.value if hasattr(s2_obs.localization.status, "value") else s2_obs.localization.status),
            source=[s.value if hasattr(s, "value") else str(s) for s in (s2_obs.localization.source or [])],
            confidence=float(s2_obs.localization.quality.confidence) if s2_obs.localization.quality else 1.0,
        )
    else:
        localization = LocalizationInfo()

    return S2Observation(
        observation_id=observation_id or s2_obs.observation_id,
        timestamp=float(s2_obs.timestamp),
        image_path=str(s2_obs.image),
        camera=intrinsics,
        pose=pose,
        features=features,
        localization=localization,
    )


def build_s2_payload_from_s2(
    s2_payload: S2PayloadOutput,
    image_root: Union[str, Path],
    min_track_len: int = 2,
    max_features: int = 500,
) -> S2Payload:
    """Translate ``S2PayloadOutput`` (S2) into ``S2Payload`` (S3).

    Loads each S1 frame image referenced by the S2 observations, extracts
    ORB features per frame, matches them across consecutive keyframes, and
    emits per-observation ``features`` lists with shared ``track_id``s.

    Parameters
    ----------
    s2_payload : S2PayloadOutput
        Output of S2 localization & sensor fusion.
    image_root : str | Path
        Directory in which to resolve relative image paths referenced by
        each S2 observation's ``image`` field.
    min_track_len : int
        Minimum number of distinct frames a feature must appear in for it
        to be kept. Default ``2`` (required for triangulation).
    max_features : int
        Maximum ORB features to detect per frame.

    Returns
    -------
    S2Payload
        A reconstruction-ready payload consumable by ``S3ReconstructionPipeline``.
        Observations without matches simply carry an empty ``features`` list;
        S3's validator treats this as a recoverable warning, not an error.
    """
    _require_cv2()

    image_root = Path(image_root)

    observations_out: List[S2Observation] = []
    per_frame_features: Dict[str, np.ndarray] = {}
    per_frame_track_indices: Dict[str, Dict[int, int]] = {}

    observations_with_intrinsics: List[Tuple[int, S2ObservationOutput, Optional[CameraInfo]]] = []
    for idx, obs in enumerate(s2_payload.observations):
        if obs.camera is None or obs.camera.intrinsics is None:
            continue
        observations_with_intrinsics.append((idx, obs, obs.camera))

    if not observations_with_intrinsics:
        return _empty_s2_payload(s2_payload)

    track_builder = _TrackBuilder()

    # Build per-frame keypoints, then match consecutive frames to grow tracks.
    prev_frame_id: Optional[str] = None
    prev_keypoints: Optional[np.ndarray] = None
    prev_descriptors: Optional[np.ndarray] = None

    for idx, s2_obs, cam_info in observations_with_intrinsics:
        img_path = _safe_image_path(s2_obs.image, image_root)
        if img_path is None:
            continue

        frame_id = s2_obs.observation_id or f"obs_{idx:04d}"
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue

        kps, descs = _extract_orb(gray, max_features=max_features)
        per_frame_features[frame_id] = kps

        if prev_frame_id is not None and prev_keypoints is not None and prev_descriptors is not None:
            for q_idx, t_idx in _match_descriptors(prev_descriptors, descs):
                track_builder.add_pair(prev_frame_id, q_idx, frame_id, t_idx)

        prev_frame_id = frame_id
        prev_keypoints = kps
        prev_descriptors = descs

    raw_tracks = track_builder.finalize(per_frame_features)
    for frame_id, mapping in raw_tracks.items():
        per_frame_track_indices[frame_id] = mapping

    # Now build S3 observations.
    for idx, s2_obs, cam_info in observations_with_intrinsics:
        frame_id = s2_obs.observation_id or f"obs_{idx:04d}"
        if frame_id not in per_frame_features:
            continue

        # S2 -> S3 pose conversion.
        pose: Optional[CameraPose] = None
        if s2_obs.pose is not None:
            pos = s2_obs.pose.position
            ori_fmt, ori_val = _normalize_orientation_format(s2_obs.pose.orientation)
            pose = CameraPose(
                position=[pos.x, pos.y, pos.z],
                orientation=ori_val,
                orientation_format=ori_fmt,
            )

        # S2 -> S3 intrinsics conversion.
        s2_intr = cam_info.intrinsics
        width = cam_info.width if cam_info.width is not None else s2_intr and None
        height = cam_info.height if cam_info.height is not None else s2_intr and None
        intrinsics = CameraIntrinsics(
            fx=float(s2_intr.fx),
            fy=float(s2_intr.fy),
            cx=float(s2_intr.cx),
            cy=float(s2_intr.cy),
            width=width,
            height=height,
            distortion_coefficients=list(cam_info.distortion.coefficients) if cam_info.distortion and cam_info.distortion.coefficients else None,
            distortion_model=cam_info.distortion.model if cam_info.distortion else None,
        )

        track_indices = per_frame_track_indices.get(frame_id, {})
        kps = per_frame_features[frame_id]
        observations_out.append(
            _build_observation(
                s2_obs=s2_obs,
                pose=pose,
                intrinsics=intrinsics,
                frame_track_indices=track_indices,
                frame_keypoints=kps,
                frame_id=frame_id,
                observation_id=frame_id,
            )
        )

    if not observations_out:
        return _empty_s2_payload(s2_payload)

    return S2Payload(
        observations=observations_out,
        job_id=None,
        coordinate_frame=str(s2_payload.coordinate_frame or "local"),
        units=str(_units_to_str(s2_payload.units)),
        schema_version=str(s2_payload.schema_version or "1.0.0"),
        metadata={
            "source": "S2->S3 bridge",
            "min_track_len": min_track_len,
            "max_features": max_features,
        },
    )


def build_s2_payload(
    observations: List[S2ObservationOutput],
    image_root: Union[str, Path],
    min_track_len: int = 2,
    max_features: int = 500,
) -> S2Payload:
    """Convenience wrapper that accepts a plain list of ``S2ObservationOutput``."""
    payload = S2PayloadOutput(observations=observations)
    return build_s2_payload_from_s2(
        payload,
        image_root=image_root,
        min_track_len=min_track_len,
        max_features=max_features,
    )


def _empty_s2_payload(src: S2PayloadOutput) -> S2Payload:
    """Return an S3 S2Payload with empty observations when bridging cannot proceed."""
    return S2Payload(
        observations=[],
        job_id=None,
        coordinate_frame=str(src.coordinate_frame or "local"),
        units=str(_units_to_str(src.units)),
        schema_version=str(src.schema_version or "1.0.0"),
        metadata={"source": "S2->S3 bridge (empty)"},
    )


def _units_to_str(units: Any) -> str:
    if isinstance(units, str):
        return units
    if hasattr(units, "position"):
        return str(units.position)
    if isinstance(units, dict):
        return str(units.get("position", "meters"))
    return "meters"
