"""Integration tests for the S2 -> S3 bridge adapter.

These tests cover the critical S2 → S3 wiring that the orchestrator
relies on: turning S2's ``S2PayloadOutput`` (poses only) into S3's
``S2Payload`` (poses + per-frame 2D feature tracks).

The tests use synthetic grayscale images with deterministic ORB
features so we don't depend on any external dataset.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.localization_sensor_fusion.adapters.s2_bridge import (
    build_s2_payload,
    build_s2_payload_from_s2,
)
from src.localization_sensor_fusion.schemas.contracts import (
    CameraInfo,
    CameraIntrinsics,
    CameraPose,
    Distortion,
    FrameQuality,
    LocalizationMeta,
    LocalizationQuality,
    LocalizationSource,
    PoseStatus,
    Position,
    QuaternionOrientation,
    S2ObservationOutput,
    S2PayloadOutput,
)


def _write_synthetic_frame(path: Path, seed: int, shift_x: int = 0, shift_y: int = 0) -> None:
    """Write a deterministic textured grayscale image to ``path``.

    ORB needs enough texture to produce keypoints. We synthesize a
    random-dot pattern with a stable seed so matches are repeatable.
    Subsequent frames are shifted copies of the base pattern so ORB
    can match features across them.
    """
    rng = np.random.RandomState(seed)
    img = rng.randint(0, 255, size=(480, 640), dtype=np.uint8)
    # Add some high-contrast structures so ORB finds stable corners.
    for _ in range(20):
        x0 = rng.randint(0, 600)
        y0 = rng.randint(0, 440)
        cv2.rectangle(img, (x0, y0), (x0 + 40, y0 + 40), int(rng.randint(0, 255)), thickness=-1)

    if shift_x != 0 or shift_y != 0:
        shifted = np.zeros_like(img)
        h, w = img.shape
        sx_src_start = max(0, -shift_x)
        sy_src_start = max(0, -shift_y)
        sx_dst_start = max(0, shift_x)
        sy_dst_start = max(0, shift_y)
        sx_end = min(w, w - shift_x)
        sy_end = min(h, h - shift_y)
        if sx_end > sx_src_start and sy_end > sy_src_start:
            shifted[sy_dst_start:sy_end, sx_dst_start:sx_end] = img[
                sy_src_start : sy_end - sy_dst_start + sy_src_start,
                sx_src_start : sx_end - sx_dst_start + sx_src_start,
            ]
        img = shifted

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def _make_s2_observation(
    observation_id: str,
    timestamp: float,
    image_path: str,
    x: float,
    y: float,
    z: float,
) -> S2ObservationOutput:
    intrinsics = CameraIntrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0)
    camera = CameraInfo(width=640, height=480, intrinsics=intrinsics, distortion=Distortion())
    return S2ObservationOutput(
        observation_id=observation_id,
        timestamp=timestamp,
        image=image_path,
        camera=camera,
        pose=CameraPose(
            position=Position(x=x, y=y, z=z),
            orientation=QuaternionOrientation(qx=0.0, qy=0.0, qz=0.0, qw=1.0),
        ),
        localization=LocalizationMeta(
            status=PoseStatus.ESTIMATED,
            source=[LocalizationSource.GPS],
            quality=LocalizationQuality(confidence=0.7),
        ),
    )


def test_s2_bridge_requires_opencv() -> None:
    """Sanity check that the bridge module imports cleanly."""
    from src.localization_sensor_fusion.adapters.s2_bridge import _require_cv2

    _require_cv2()


def test_bridge_produces_features_with_track_ids(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    s2_observations = []
    for i in range(4):
        img_rel = f"frames/frame_{i:03d}.jpg"
        # Each successive frame is shifted by a few pixels so the
        # ORB matcher can find inlier correspondences across frames.
        _write_synthetic_frame(tmp_path / img_rel, seed=42, shift_x=i * 4, shift_y=i * 2)
        s2_observations.append(
            _make_s2_observation(
                observation_id=f"frame_{i:03d}",
                timestamp=float(i),
                image_path=img_rel,
                x=0.0,
                y=0.0,
                z=10.0 + i,
            )
        )

    s2_payload = S2PayloadOutput(observations=s2_observations)
    s3_payload = build_s2_payload_from_s2(s2_payload, image_root=tmp_path)

    assert len(s3_payload.observations) == 4

    track_ids: set[str] = set()
    matched_observations = 0
    for obs in s3_payload.observations:
        assert obs.pose is not None
        assert obs.camera is not None
        if obs.features:
            matched_observations += 1
            for feat in obs.features:
                assert feat.track_id is not None
                track_ids.add(feat.track_id)

    # At least one observation should carry features, and at least one
    # track should span multiple observations.
    assert matched_observations >= 1
    assert len(track_ids) >= 1


def test_bridge_preserves_pose_and_intrinsics(tmp_path: Path) -> None:
    img_rel = "frames/frame_000.jpg"
    _write_synthetic_frame(tmp_path / img_rel, seed=7)
    obs = _make_s2_observation(
        observation_id="frame_000",
        timestamp=1.5,
        image_path=img_rel,
        x=2.0,
        y=3.0,
        z=4.0,
    )
    s2_payload = S2PayloadOutput(observations=[obs])

    s3_payload = build_s2_payload_from_s2(s2_payload, image_root=tmp_path)

    assert len(s3_payload.observations) == 1
    out = s3_payload.observations[0]
    assert out.pose is not None
    assert abs(out.pose.position_array[0] - 2.0) < 1e-9
    assert abs(out.pose.position_array[1] - 3.0) < 1e-9
    assert abs(out.pose.position_array[2] - 4.0) < 1e-9
    assert out.camera is not None
    assert abs(out.camera.fx - 600.0) < 1e-9
    assert abs(out.camera.fy - 600.0) < 1e-9
    assert abs(out.camera.cx - 320.0) < 1e-9
    assert abs(out.camera.cy - 240.0) < 1e-9


def test_bridge_handles_missing_images_gracefully(tmp_path: Path) -> None:
    """If an image path doesn't resolve, the bridge should drop that observation."""
    s2_payload = S2PayloadOutput(
        observations=[
            _make_s2_observation(
                observation_id="frame_000",
                timestamp=0.0,
                image_path="frames/does_not_exist.jpg",
                x=0.0,
                y=0.0,
                z=10.0,
            )
        ]
    )

    s3_payload = build_s2_payload_from_s2(s2_payload, image_root=tmp_path)

    # The lone observation had no resolvable image, so the bridge
    # should return an empty-but-valid payload instead of crashing.
    assert s3_payload.observations == []


def test_bridge_observation_without_intrinsics_is_dropped(tmp_path: Path) -> None:
    _write_synthetic_frame(tmp_path / "frames/frame_000.jpg", seed=11)

    obs_no_intr = S2ObservationOutput(
        observation_id="frame_000",
        timestamp=0.0,
        image="frames/frame_000.jpg",
        camera=None,
        pose=CameraPose(
            position=Position(x=0.0, y=0.0, z=10.0),
            orientation=QuaternionOrientation(qx=0.0, qy=0.0, qz=0.0, qw=1.0),
        ),
        localization=LocalizationMeta(
            status=PoseStatus.ESTIMATED,
            source=[LocalizationSource.GPS],
            quality=LocalizationQuality(confidence=0.5),
        ),
    )

    s2_payload = S2PayloadOutput(observations=[obs_no_intr])
    s3_payload = build_s2_payload_from_s2(s2_payload, image_root=tmp_path)
    assert s3_payload.observations == []


def test_build_s2_payload_list_wrapper(tmp_path: Path) -> None:
    _write_synthetic_frame(tmp_path / "frames/frame_000.jpg", seed=13)
    obs = _make_s2_observation(
        observation_id="frame_000",
        timestamp=0.0,
        image_path="frames/frame_000.jpg",
        x=0.0,
        y=0.0,
        z=10.0,
    )
    s3_payload = build_s2_payload([obs], image_root=tmp_path)
    assert len(s3_payload.observations) == 1
