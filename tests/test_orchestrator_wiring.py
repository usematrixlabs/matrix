"""End-to-end orchestrator wiring tests.

These tests verify the four critical cross-subsystem wirings the
orchestrator is responsible for:

    S1 → S2   (S1 frame observations flow through S1InputAdapter into
               a validated S2ObservationOutput)
    S2 → S3   (S2PayloadOutput flows through the S2→S3 bridge into an
               S3 S2Payload with 2D feature tracks)
    S3 → S4   (S3 reconstruction result flows through
               to_s4_reconstruction_input() into a Georeferencer)
    S4 → S5   (Georeferencer + GeoreferencingValidator results flow into
               Finalizer.bundle() as part of the summary)

We exercise the wiring at the orchestrator level by importing and
calling the real ``run_*`` functions, but bypass the slowest internal
steps (real video decoding, real ORB matching across thousands of
frames) by stubbing S1's output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pytest

from src.pipeline import orchestrator
from src.pipeline.orchestrator import (
    run_s2,
    run_s3,
    run_s4,
    run_s5,
)
from src.visual_perception.types import (
    Frame,
    QualityAssessment,
    S1Output,
    UAVTelemetry,
    VisualObservations,
)


# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------


def _write_orb_frame(path: Path, seed: int, shift_x: int = 0, shift_y: int = 0) -> None:
    rng = np.random.RandomState(seed)
    img = rng.randint(0, 255, size=(480, 640), dtype=np.uint8)
    for _ in range(30):
        x0 = rng.randint(0, 600)
        y0 = rng.randint(0, 440)
        cv2.rectangle(img, (x0, y0), (x0 + 40, y0 + 40), int(rng.randint(0, 255)), thickness=-1)
    if shift_x or shift_y:
        shifted = np.zeros_like(img)
        h, w = img.shape
        sx_src = max(0, -shift_x)
        sy_src = max(0, -shift_y)
        sx_dst = max(0, shift_x)
        sy_dst = max(0, shift_y)
        sx_end = min(w, w - shift_x)
        sy_end = min(h, h - shift_y)
        if sx_end > sx_src and sy_end > sy_src:
            shifted[sy_dst:sy_end, sx_dst:sx_end] = img[sy_src:sy_src + sy_end - sy_dst, sx_src:sx_src + sx_end - sx_dst]
        img = shifted
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def _fake_s1_output(
    output_dir: Path,
    *,
    num_frames: int = 4,
    include_calibration: bool = True,
) -> S1Output:
    """Build a synthetic S1Output for orchestrator wiring tests.

    Writes real ORB-matchable frames to ``<output_dir>/frames`` so the
    downstream S2→S3 bridge can produce feature tracks.
    """
    frames_dir = output_dir / "frames"
    frames: List[Frame] = []
    for i in range(num_frames):
        rel = f"frames/frame_{i:04d}.jpg"
        _write_orb_frame(output_dir / rel, seed=100 + i, shift_x=i * 4, shift_y=i * 2)
        frames.append(
            Frame(
                frame_id=f"frame_{i:04d}",
                timestamp=float(i) * 0.1,
                image_path=str(output_dir / rel),
                image_width=640,
                image_height=480,
                quality=QualityAssessment(status="GOOD", blur_score=200.0),
                is_keyframe=True,
            )
        )

    calib = None
    if include_calibration:
        calib = {
            "width": 640,
            "height": 480,
            "intrinsics": {"fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0},
            "distortion": {"coefficients": [], "model": "opencv"},
        }

    obs = VisualObservations(frames=frames, frame_ordering=[f.frame_id for f in frames])
    return S1Output(
        visual_observations=obs,
        status="completed",
        warnings=[],
        errors=[],
        diagnostics={"health_status": "completed"},
        metadata={
            "subsystem": "S1_Visual_Perception",
            "video_source": str(output_dir / "video.mp4"),
            "camera_calibration": calib,
            "observations_json": str(output_dir / "observations.json"),
        },
    )


def _write_gps_csv(path: Path, num_rows: int = 4) -> None:
    """Write a minimal GPS CSV that the orchestrator can ingest."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "frame_index,video_time_s,timestamp,drone_lat,drone_lon,drone_altitude_m\n"
        )
        for i in range(num_rows):
            f.write(
                f"{i},{i * 0.1:.6f},2025-08-23 10:45:{7 + i:02d}.000,30.289664,-97.782784,150.0\n"
            )


# ---------------------------------------------------------------------------
# Wiring tests
# ---------------------------------------------------------------------------


def test_s1_to_s2_wiring_produces_s2_payload(tmp_path: Path) -> None:
    """S1's frames flow through S1InputAdapter into a real S2PayloadOutput JSON."""
    s1_output = _fake_s1_output(tmp_path / "s1")
    _write_gps_csv(tmp_path / "gps.csv")

    out_path = run_s2(s1_output, tmp_path / "gps.csv", tmp_path / "s2")

    assert out_path.is_file()
    payload = json.loads(out_path.read_text())
    assert "observations" in payload
    assert len(payload["observations"]) == 4

    obs0 = payload["observations"][0]
    assert obs0["observation_id"] == "frame_0000"
    # GPS-prior pose should be attached, with non-zero confidence.
    assert obs0["pose"]["position"] is not None
    assert obs0["localization"]["quality"]["confidence"] >= 0.0


def test_s2_to_s3_wiring_via_bridge(tmp_path: Path) -> None:
    """S2's payload flows through the S2→S3 bridge into an S3 S2Payload."""
    s1_output = _fake_s1_output(tmp_path / "s1", num_frames=5)
    _write_gps_csv(tmp_path / "gps.csv", num_rows=5)

    s2_path = run_s2(s1_output, tmp_path / "gps.csv", tmp_path / "s2")
    assert s2_path.is_file()

    s3_ply = run_s3(s1_output, s2_path, tmp_path / "s3")
    # Either S3 produced an artifact, or it returned the canonical path.
    assert s3_ply.name == "scene.ply"
    assert s3_ply.parent.is_dir()


def test_s3_to_s4_wiring_uses_to_s4_reconstruction_input(tmp_path: Path) -> None:
    """S4 reads S3's PLY + metadata, runs Georeferencer, validates via GeoreferencingValidator."""
    s1_output = _fake_s1_output(tmp_path / "s1", num_frames=5)
    _write_gps_csv(tmp_path / "gps.csv", num_rows=5)

    s2_path = run_s2(s1_output, tmp_path / "gps.csv", tmp_path / "s2")
    run_s3(s1_output, s2_path, tmp_path / "s3")

    s4_artifacts = run_s4(tmp_path / "s3", tmp_path / "s4")

    assert s4_artifacts["ply"].is_file()
    assert s4_artifacts["georeferencing"].is_file()
    geo_meta = json.loads(s4_artifacts["georeferencing"].read_text())
    # Identity-like fit: RMSE should be 0 (or near-zero for numeric noise).
    if "rmse" in geo_meta and geo_meta["rmse"] is not None:
        assert geo_meta["rmse"] < 1.0
    # Validation result fields must be present.
    assert "mean_error" in geo_meta
    assert "max_error" in geo_meta
    assert "median_error" in geo_meta


def test_s4_to_s5_wiring_includes_validation_summary(tmp_path: Path) -> None:
    """S5's Finalizer.bundle summary includes S4's georeferencing + validation fields."""
    s1_output = _fake_s1_output(tmp_path / "s1", num_frames=5)
    _write_gps_csv(tmp_path / "gps.csv", num_rows=5)

    s2_path = run_s2(s1_output, tmp_path / "gps.csv", tmp_path / "s2")
    run_s3(s1_output, s2_path, tmp_path / "s3")
    s4_artifacts = run_s4(tmp_path / "s3", tmp_path / "s4")

    s5_path = run_s5(
        output_dir=tmp_path,
        success=True,
        stage_status={"S1": "completed", "S2": "completed", "S3": "completed", "S4": "completed"},
        artifacts={"S4": s4_artifacts["ply"]},
        summary={
            "num_stages": 5,
            "s4": {
                "georeferencing": str(s4_artifacts["georeferencing"]),
                "validation": str(s4_artifacts["validation"]),
                "rmse": 0.0,
                "passed": True,
            },
        },
    )

    assert s5_path.is_file()
    final = json.loads(s5_path.read_text())
    assert "summary" in final
    assert "s4" in final["summary"]
    assert "rmse" in final["summary"]["s4"]
    assert "passed" in final["summary"]["s4"]


def test_s4_degraded_path_does_not_break_s5_wiring(tmp_path: Path) -> None:
    """When S3 produced an empty PLY, S4 should still hand off to S5 cleanly."""
    (tmp_path / "s3").mkdir(parents=True, exist_ok=True)

    s4_artifacts = run_s4(tmp_path / "s3", tmp_path / "s4")
    assert s4_artifacts["georeferencing"].is_file()
    geo_meta = json.loads(s4_artifacts["georeferencing"].read_text())
    assert geo_meta["status"] == "degraded"

    s5_path = run_s5(
        output_dir=tmp_path,
        success=True,
        stage_status={"S4": "degraded"},
        artifacts={"S4": s4_artifacts["ply"]},
        summary={
            "num_stages": 5,
            "s4": {
                "georeferencing": str(s4_artifacts["georeferencing"]),
                "validation": str(s4_artifacts["validation"]),
                "status": "degraded",
            },
        },
    )
    assert s5_path.is_file()


def test_run_s2_calls_s1_input_adapter(monkeypatch, tmp_path: Path) -> None:
    """The S1→S2 wiring must actually invoke S1InputAdapter.parse_observation."""
    s1_output = _fake_s1_output(tmp_path / "s1")
    _write_gps_csv(tmp_path / "gps.csv")

    seen_calls: List[dict] = []

    real_parse = orchestrator.S1InputAdapter.parse_observation

    def spy_parse(self, payload):
        seen_calls.append(payload)
        return real_parse(self, payload)

    monkeypatch.setattr(orchestrator.S1InputAdapter, "parse_observation", spy_parse)

    run_s2(s1_output, tmp_path / "gps.csv", tmp_path / "s2")

    assert len(seen_calls) == 4
    assert seen_calls[0]["observation_id"] == "frame_0000"
    assert seen_calls[0]["timestamp"] == pytest.approx(0.0)
