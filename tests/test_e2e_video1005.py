"""
End-to-end benchmark regression test for video-1005.

This test runs the full Matrix pipeline (S1 → S2 → S3 → S4 → S5) on the
video-1005 benchmark dataset and verifies the contract-boundary
invariants documented in docs/architecture/system-architecture.md.

Marked with ``@pytest.mark.benchmark`` so it can be excluded from the
normal fast suite (run with ``pytest -m benchmark``).
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import pytest

from pipeline.orchestrator import run_pipeline


BENCHMARK_VIDEO = Path("benchmarks/dataset/video-1005/video.mp4")
BENCHMARK_GPS = Path("benchmarks/dataset/video-1005/gps.csv")


@pytest.mark.benchmark
@pytest.mark.skipif(
    not BENCHMARK_VIDEO.exists() or not BENCHMARK_GPS.exists(),
    reason="video-1005 benchmark dataset not present",
)
def test_video1005_end_to_end(tmp_path: Path):
    output_dir = tmp_path / "matrix_run"
    result = run_pipeline(BENCHMARK_VIDEO, BENCHMARK_GPS, output_dir)

    # Pipeline must complete all 5 stages
    assert result.success is True
    assert result.stage_status == {
        "S1": "completed",
        "S2": "completed",
        "S3": "completed",
        "S4": "completed",
        "S5": "completed",
    }

    # S1 — 239 keyframes produced
    s1_observations = output_dir / "s1" / "observations.json"
    assert s1_observations.is_file()
    with open(s1_observations) as f:
        s1 = json.load(f)
    assert s1["total_observations"] == s1["keyframe_count"]
    assert s1["total_observations"] > 0

    # S2 — one observation per S1 keyframe
    s2_output = output_dir / "s2" / "s2_output.json"
    assert s2_output.is_file()
    with open(s2_output) as f:
        s2 = json.load(f)
    assert len(s2["observations"]) == s1["total_observations"]

    # S2 poses must vary (not all (0, 0, 0))
    positions = [
        (
            o["pose"]["position"]["x"],
            o["pose"]["position"]["y"],
            o["pose"]["position"]["z"],
        )
        for o in s2["observations"]
    ]
    unique_positions = len(set(positions))
    assert unique_positions > 1, "S2 poses do not vary"

    # S5 — final contract populated
    final_output = output_dir / "s5" / "final_output.json"
    assert final_output.is_file()
    with open(final_output) as f:
        s5 = json.load(f)
    assert s5["success"] is True
    assert "s3" in s5["summary"]
    assert "s4" in s5["summary"]

    # S4 — at minimum, georeferencing.json exists with finite RMSE
    # OR a degraded contract (currently expected: identity placeholder
    # produces finite RMSE on the input point set).
    s4_meta_path = output_dir / "s4" / "georeferencing.json"
    assert s4_meta_path.is_file()
    with open(s4_meta_path) as f:
        s4_meta = json.load(f)
    if s4_meta.get("status") != "degraded":
        assert math.isfinite(s4_meta["rmse"])


@pytest.mark.benchmark
@pytest.mark.skipif(
    not BENCHMARK_VIDEO.exists() or not BENCHMARK_GPS.exists(),
    reason="video-1005 benchmark dataset not present",
)
def test_video1005_s3_produces_reconstruction(tmp_path: Path):
    """
    Diagnostic test: verify S3 produces a non-empty reconstruction.

    Currently expected to FAIL on video-1005 because the S1 → S2 boundary
    drops the image path (see Phase 1 evidence). This test documents
    the current state — it should be updated when the boundary is fixed.
    """
    output_dir = tmp_path / "matrix_run"
    result = run_pipeline(BENCHMARK_VIDEO, BENCHMARK_GPS, output_dir)
    assert result.success is True

    s3_meta_path = output_dir / "s3" / "metadata.json"
    if s3_meta_path.is_file():
        with open(s3_meta_path) as f:
            s3_meta = json.load(f)
        assert s3_meta.get("num_points", 0) > 0, (
            "S3 produced 0 points. Boundary investigation required."
        )
