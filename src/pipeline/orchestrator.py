"""Matrix end-to-end pipeline orchestrator (sealed).

The orchestrator is the **only** module in the Matrix codebase that
imports across subsystem boundaries. It owns:

- Execution order
- Cross-subsystem data movement
- Adaptation between adjacent contracts (``S1Contract`` → ``S2Contract``
  → ``S3Contract`` → ``S4Contract`` → ``S5Contract``)
- Artifact location conventions
- Status / failure propagation

It does **not** import from any subsystem's ``_internal/`` namespace.

Flow:

    video.mp4 ──► S1 ──► S2 ◄── gps.csv ──► S3 ──► S4 ──► S5 ──► final

Outputs are written to ``<output_dir>/sN/`` per subsystem; S5 bundles a
top-level ``final_output.json`` summary into ``<output_dir>/s5/``.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from application_deployment import S5Contract, run_s5
from georeferencing_validation import S4Contract, run_s4
from localization_sensor_fusion import S2Contract, run_s2
from reconstruction import S3Contract, run_s3
from visual_perception import S1Output, run_s1, s1_output_to_contract


PIPELINE_TAG = "[MATRIX]"


@dataclass
class PipelineResult:
    """Outcome of a single end-to-end pipeline run."""

    success: bool
    output_dir: Path
    final_output: Optional[Path] = None
    failed_stage: Optional[str] = None
    error: Optional[str] = None
    stage_status: Optional[Dict[str, str]] = None


def _log(msg: str) -> None:
    print(f"{PIPELINE_TAG} {msg}", flush=True)


def _stage_log(stage: str, msg: str) -> None:
    print(f"[{stage}] {msg}", flush=True)


def run_pipeline(
    video_path: Union[str, Path],
    gps_path: Union[str, Path],
    output_dir: Union[str, Path],
) -> PipelineResult:
    """Run the full Matrix pipeline on a single video + GPS pair.

    Returns a :class:`PipelineResult`. On any stage failure the pipeline
    stops, preserves all previously written outputs under ``output_dir``,
    and returns ``success=False`` with ``failed_stage`` and ``error`` set.
    """
    video_path = Path(video_path)
    gps_path = Path(gps_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _log("Starting pipeline")
    _log(f"video: {video_path}")
    _log(f"gps:   {gps_path}")
    _log(f"out:   {output_dir}")

    stage_status: Dict[str, str] = {}
    artifacts: Dict[str, str] = {}
    failed_stage: Optional[str] = None

    def _ok(stage: str, artifact: Optional[Union[Path, str]] = None) -> None:
        _stage_log(stage, "✓ Complete")
        stage_status[stage] = "completed"
        if artifact is not None:
            artifacts[stage] = str(artifact)

    try:
        _log("S1 — Visual Perception")
        failed_stage = "S1"
        s1_output: S1Output = run_s1(video_path, output_dir / "s1")
        # The orchestrator is the **only** place that constructs the
        # S1 → S2 boundary contract.
        s1_contract = s1_output_to_contract(s1_output)
        s1_artifact = (s1_output.metadata or {}).get("observations_json")
        _ok("S1", s1_artifact or str(output_dir / "s1"))
        failed_stage = None

        _log("S2 — Localization & Sensor Fusion")
        failed_stage = "S2"
        s2_contract: S2Contract = run_s2(
            s1_contract=s1_contract,
            gps_path=gps_path,
            output_dir=output_dir / "s2",
        )
        _ok("S2", str(output_dir / "s2" / "s2_output.json"))
        failed_stage = None

        _log("S3 — 3D Reconstruction")
        failed_stage = "S3"
        # S3 reads the S1 frame images referenced by S2.
        s1_root = output_dir / "s1"
        s3_contract: S3Contract = run_s3(
            s2_contract=s2_contract,
            image_root=s1_root,
            output_dir=output_dir / "s3",
        )
        _ok("S3", s3_contract.artifact_paths.get("ply"))
        failed_stage = None

        _log("S4 — Georeferencing & Validation")
        failed_stage = "S4"
        s4_contract: S4Contract = run_s4(
            s3_contract=s3_contract,
            output_dir=output_dir / "s4",
        )
        _ok("S4", s4_contract.artifact_paths.get("ply"))
        # Inline S4's RMSE / pass status into the S5 summary if present.
        s4_meta_path = s4_contract.artifact_paths.get("georeferencing")
        s4_meta: Dict[str, Any] = {}
        if s4_meta_path and Path(s4_meta_path).is_file():
            try:
                with open(s4_meta_path, "r", encoding="utf-8") as f:
                    s4_meta = json.load(f)
            except (OSError, json.JSONDecodeError):
                s4_meta = {}
        failed_stage = None

        _log("S5 — Application & Deployment")
        failed_stage = "S5"
        s5_summary = {
            "num_stages": 5,
            "s3": {
                "scene_id": s3_contract.scene_id,
                "status": s3_contract.status,
                "num_points": (s3_contract.point_cloud or {}).get("num_points", 0),
            },
            "s4": {
                "georeferencing": s4_contract.artifact_paths.get("georeferencing"),
                "validation": s4_contract.artifact_paths.get("validation"),
                "rmse": s4_meta.get("rmse"),
                "mean_error": s4_meta.get("mean_error"),
                "max_error": s4_meta.get("max_error"),
                "median_error": s4_meta.get("median_error"),
                "passed": s4_meta.get("passed"),
                "status": s4_contract.status,
            },
        }
        s5_contract: S5Contract = run_s5(
            s4_contract=s4_contract,
            output_dir=output_dir,
            success=True,
            stage_status=stage_status,
            artifacts=artifacts,
            summary=s5_summary,
        )
        _ok("S5", str(output_dir / "s5" / "final_output.json"))
        failed_stage = None

    except Exception as exc:  # noqa: BLE001 - top-level pipeline boundary
        stage_label = failed_stage or "UNKNOWN"
        _stage_log(stage_label, f"✗ Failed: {exc}")
        stage_status[stage_label] = "failed"
        _log(f"Pipeline stopped at {stage_label}")
        _log(f"Partial results preserved at:\n         {output_dir}")
        sys.stderr.write(traceback.format_exc())
        return PipelineResult(
            success=False,
            output_dir=output_dir,
            failed_stage=stage_label,
            error=str(exc),
            stage_status=stage_status,
        )

    _log("Pipeline complete")
    return PipelineResult(
        success=True,
        output_dir=output_dir,
        final_output=output_dir / "s5" / "final_output.json",
        stage_status=stage_status,
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Matrix — end-to-end UAV video to 3D geospatial pipeline"
    )
    parser.add_argument("--video", required=True, type=Path, help="Path to UAV video (.mp4)")
    parser.add_argument("--gps", required=True, type=Path, help="Path to GPS CSV")
    parser.add_argument("--output", required=True, type=Path, help="Output directory for this run")
    args = parser.parse_args(argv)

    result = run_pipeline(args.video, args.gps, args.output)

    print(json.dumps(
        {
            "success": result.success,
            "output_dir": str(result.output_dir),
            "final_output": str(result.final_output) if result.final_output else None,
            "failed_stage": result.failed_stage,
            "error": result.error,
            "stage_status": result.stage_status,
        },
        indent=2,
    ))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
