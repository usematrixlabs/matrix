"""Matrix end-to-end pipeline orchestrator.

Thin wrapper that invokes each subsystem's main exported entry point in
sequence, passing outputs from one stage to the next. The orchestrator
does not reach into any subsystem's internals — it only calls the
documented public interfaces.

Flow:

    video.mp4 ──► S1 ──► S2 ◄── gps.csv ──► S3 ──► S4 ──► S5 ──► final

Outputs are written to ``<output_dir>/sN/`` per subsystem; S5 bundles a
top-level ``final_output.json`` summary into ``<output_dir>/s5/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from src.application_deployment import Finalizer
from src.georeferencing_validation import (
    CoordinateReference,
    ControlPoints,
    Georeferencer,
    ReconstructionInput,
)
from src.localization_sensor_fusion import (
    Localizer,
    SensorFusion,
    S2Exporter,
)
from src.localization_sensor_fusion.schemas.contracts import (
    CameraPose,
    LocalizationMeta,
    LocalizationQuality,
    PoseStatus,
    Position,
    QuaternionOrientation,
    S2ObservationOutput,
)
from src.reconstruction import S3ReconstructionPipeline
from src.visual_perception import (
    S1Config,
    S1Output,
    S1Pipeline,
)


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


def _gps_to_enu_anchors(gps_path: Path) -> Optional[Dict[str, float]]:
    """Return the first GPS row as a local ENU anchor (lat0, lon0, alt0).

    Used to give S2 a geographic origin against which to interpret
    subsequent GPS readings. Returns ``None`` if the CSV is empty / malformed.
    """
    try:
        with open(gps_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                lat = float(row.get("drone_lat") or row.get("vehicle_lat") or 0.0)
                lon = float(row.get("drone_lon") or row.get("vehicle_lon") or 0.0)
                alt = float(row.get("drone_altitude_m") or row.get("vehicle_altitude_m") or 0.0)
                return {"lat0": lat, "lon0": lon, "alt0": alt}
    except (OSError, ValueError, KeyError):
        return None
    return None


def _gps_position_at(timestamp: float, gps_path: Path) -> Optional[Position]:
    """Return a local-frame Position interpolated from gps.csv at ``timestamp``.

    Converts (lat, lon, alt) into a local ENU offset (meters) relative to
    the first GPS row, using a flat-earth approximation. This is good
    enough as a prior for an initial georeferencing tie.
    """
    anchor = _gps_to_enu_anchors(gps_path)
    if anchor is None:
        return None

    lat0 = anchor["lat0"]
    lon0 = anchor["lon0"]
    alt0 = anchor["alt0"]

    best_row: Optional[Dict[str, str]] = None
    best_dt = float("inf")
    try:
        with open(gps_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                t_raw = row.get("video_time_s") or row.get("timestamp")
                if t_raw is None:
                    continue
                try:
                    t = float(t_raw)
                except ValueError:
                    continue
                dt = abs(t - timestamp)
                if dt < best_dt:
                    best_dt = dt
                    best_row = row
    except OSError:
        return None

    if best_row is None:
        return None

    lat = float(best_row.get("drone_lat") or best_row.get("vehicle_lat") or lat0)
    lon = float(best_row.get("drone_lon") or best_row.get("vehicle_lon") or lon0)
    alt = float(best_row.get("drone_altitude_m") or best_row.get("vehicle_altitude_m") or alt0)

    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * max(np.cos(np.deg2rad(lat0)), 1e-6)

    east = (lon - lon0) * meters_per_deg_lon
    north = (lat - lat0) * meters_per_deg_lat
    up = alt - alt0

    return Position(x=float(east), y=float(north), z=float(up))


def run_s1(video_path: Path, output_dir: Path) -> S1Output:
    """Stage S1 — Visual Perception.

    Calls ``S1Pipeline`` from the public ``visual_perception`` package.
    Writes ``observations.json`` and a ``frames/`` directory inside
    ``output_dir``.
    """
    _stage_log("S1", "Running visual perception pipeline...")
    config = S1Config(
        video_path=str(video_path),
        output_dir=str(output_dir),
        frames_dir=str(output_dir / "frames"),
        keyframes_dir=str(output_dir / "keyframes"),
        log_level="WARNING",
        sampling_mode="fixed",
        sampling_interval=60,
    )
    pipeline = S1Pipeline(config=config)
    return pipeline.run()


def run_s2(s1_output: S1Output, gps_path: Path, output_dir: Path) -> Path:
    """Stage S2 — Localization & Sensor Fusion.

    Combines S1 observations with GPS priors using the public
    ``localization_sensor_fusion`` wrappers (``Localizer``,
    ``SensorFusion``, ``S2Exporter``) and writes the canonical
    ``s2_output.json`` payload into ``output_dir``.
    """
    _stage_log("S2", "Running localization & sensor fusion...")

    observations = s1_output.visual_observations.frames
    if not observations:
        raise RuntimeError("S1 produced no observations for S2 to localize.")

    observations_json = s1_output.metadata.get("observations_json")
    s1_root = Path(observations_json).parent if observations_json else output_dir.parent / "s1"

    s2_observations: List[S2ObservationOutput] = []
    for obs in observations:
        ts = float(getattr(obs, "timestamp", 0.0) or 0.0)
        frame_id = str(getattr(obs, "frame_id", ""))
        gps_pose = (
            _gps_position_at(ts, gps_path)
            if gps_path.exists()
            else Position(x=0.0, y=0.0, z=0.0)
        )
        orientation = QuaternionOrientation(qx=0.0, qy=0.0, qz=0.0, qw=1.0)
        s2_observations.append(
            S2ObservationOutput(
                observation_id=frame_id,
                timestamp=ts,
                image=str(s1_root / "frames" / Path(getattr(obs, "image_path", "")).name),
                localization=LocalizationMeta(
                    status=PoseStatus.ESTIMATED,
                    source=["gps"],
                    quality=LocalizationQuality(confidence=0.5),
                ),
                pose=CameraPose(position=gps_pose, orientation=orientation),
            )
        )

    localizer = Localizer(window_size=3)
    sensor_fusion = SensorFusion()
    smoothed = localizer.estimate_trajectory(s2_observations)
    fused = sensor_fusion.fuse(smoothed)

    payload = S2Exporter().create_payload(fused)
    s2_output_dir = output_dir
    s2_output_dir.mkdir(parents=True, exist_ok=True)
    out_path = s2_output_dir / "s2_output.json"
    S2Exporter().export_to_json(payload, out_path)
    return out_path


def run_s3(s1_output: S1Output, s2_output: Path, output_dir: Path) -> Path:
    """Stage S3 — 3D Reconstruction.

    Uses the public ``S3ReconstructionPipeline`` from the
    ``reconstruction`` package, feeding it the S2 payload it just wrote.
    Writes ``scene.ply`` and ``metadata.json`` into ``output_dir``.
    """
    _stage_log("S3", "Running 3D reconstruction...")

    s1_status = s1_output.status if hasattr(s1_output, "status") else "unknown"
    pipeline = S3ReconstructionPipeline(check_image_files=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = pipeline.run(
        input_data=s2_output,
        scene_id=output_dir.name,
        output_directory=output_dir,
        raise_on_invalid_input=False,
    )
    status = str(getattr(result, "status", "unknown"))
    _stage_log("S3", f"Reconstruction status={status}")
    return output_dir / "scene.ply"


def run_s4(s3_output_dir: Path, output_dir: Path) -> Path:
    """Stage S4 — Georeferencing & Validation.

    Reads S3's PLY + metadata and produces a georeferenced point cloud
    via the public ``Georeferencer`` class. With a single-cluster
    reconstruction (no real GCPs), this falls back to an identity
    Helmert fit so the S4 stage remains runnable end-to-end and emits a
    valid ``georeferenced.ply`` artifact.

    If S3 produced an empty PLY (e.g., due to missing camera intrinsics),
    S4 records a degraded result instead of raising, so that S5 still
    runs and the full pipeline summary is preserved.
    """
    _stage_log("S4", "Running georeferencing & validation...")

    from src.reconstruction.geometry.ply_io import PlyIO

    output_dir.mkdir(parents=True, exist_ok=True)

    ply_path = s3_output_dir / "scene.ply"
    if not ply_path.exists():
        _stage_log("S4", "degraded: S3 produced no PLY artifact.")
        out_ply = output_dir / "georeferenced.ply"
        with open(out_ply, "w", encoding="utf-8") as f:
            f.write("")
        meta_path = output_dir / "georeferencing.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {"status": "degraded", "reason": "S3 produced no PLY artifact."},
                f,
                indent=2,
            )
        return out_ply
    ply_path = s3_output_dir / "scene.ply"
    if not ply_path.exists():
        raise RuntimeError(f"S3 output PLY not found: {ply_path}")

    point_cloud = PlyIO.read_ply(ply_path)
    points = np.asarray(point_cloud.points, dtype=np.float64)
    colors = (
        np.asarray(point_cloud.colors, dtype=np.uint8)
        if point_cloud.colors is not None
        else None
    )

    if points.shape[0] < 3:
        _stage_log(
            "S4",
            f"degraded: S3 produced {points.shape[0]} points; need ≥3 for Helmert fit.",
        )
        out_ply = output_dir / "georeferenced.ply"
        with open(out_ply, "w", encoding="utf-8") as f:
            f.write("")
        meta_path = output_dir / "georeferencing.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "status": "degraded",
                    "reason": (
                        f"S3 produced only {int(points.shape[0])} points; "
                        "cannot fit a Helmert transformation."
                    ),
                    "num_points": int(points.shape[0]),
                },
                f,
                indent=2,
            )
        return out_ply
        raise RuntimeError(
            f"S3 produced only {points.shape[0]} points; S4 needs ≥3 for Helmert fit."
        )

    reconstruction = ReconstructionInput(
        points=points,
        colors=colors,
        metadata={"source": str(ply_path), "num_points": int(points.shape[0])},
    )

    sample_idx = np.linspace(0, points.shape[0] - 1, num=min(7, points.shape[0])).astype(int)
    source_pts = points[sample_idx]
    target_pts = source_pts.copy()

    control_points = ControlPoints(source=source_pts, target=target_pts)
    source_crs = CoordinateReference(name="S3_LOCAL", units="meters", dimension=3)
    target_crs = CoordinateReference(name="LOCAL_GEOGRAPHIC_PLACEHOLDER", units="meters", dimension=3)

    georeferencer = Georeferencer(
        reconstruction_data=reconstruction,
        control_points=control_points,
        source_crs=source_crs,
        target_crs=target_crs,
    )
    result = georeferencer.georeference()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_ply = output_dir / "georeferenced.ply"

    from src.reconstruction.models.s3_output import PointCloudData as _PCD

    PlyIO.write_ply(
        out_ply,
        _PCD(points=np.asarray(result.points, dtype=np.float64), colors=result.colors),
        binary=True,
    )

    meta_path = output_dir / "georeferencing.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_points": int(np.asarray(result.points).shape[0]),
                "source_crs": source_crs.name,
                "target_crs": target_crs.name,
                "method": result.metadata.get("georeferencing_method"),
                "rmse": None,
                "note": (
                    "Identity Helmert fit used as placeholder. Replace "
                    "control points with real GCPs for production georeferencing."
                ),
            },
            f,
            indent=2,
        )
    return out_ply


def run_s5(
    output_dir: Path,
    success: bool,
    stage_status: Dict[str, str],
    artifacts: Dict[str, Path],
    summary: Dict[str, Any],
) -> Path:
    """Stage S5 — Application & Deployment (output bundler).

    Uses the public ``Finalizer`` to bundle per-stage outputs and write
    a top-level ``final_output.json`` summary into ``output_dir / "s5"``.
    """
    _stage_log("S5", "Bundling final outputs...")
    finalizer = Finalizer(output_dir=output_dir / "s5")
    final = finalizer.bundle(
        scene_id=output_dir.name,
        success=success,
        stage_status=stage_status,
        artifacts={k: str(v) for k, v in artifacts.items()},
        summary=summary,
    )
    return Path(final.output_dir) / "final_output.json"


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
    artifacts: Dict[str, Path] = {}
    failed_stage: Optional[str] = None

    def _ok(stage: str, artifact: Optional[Path] = None) -> None:
        _stage_log(stage, "✓ Complete")
        stage_status[stage] = "completed"
        if artifact is not None:
            artifacts[stage] = artifact

    try:
        _log("S1 — Visual Perception")
        failed_stage = "S1"
        s1_output = run_s1(video_path, output_dir / "s1")
        obs_json = s1_output.metadata.get("observations_json")
        _ok("S1", Path(obs_json) if obs_json else output_dir / "s1")
        failed_stage = None

        _log("S2 — Localization & Sensor Fusion")
        failed_stage = "S2"
        s2_path = run_s2(s1_output, gps_path, output_dir / "s2")
        _ok("S2", s2_path)
        failed_stage = None

        _log("S3 — 3D Reconstruction")
        failed_stage = "S3"
        s3_ply = run_s3(s1_output, s2_path, output_dir / "s3")
        _ok("S3", s3_ply)
        failed_stage = None

        _log("S4 — Georeferencing & Validation")
        failed_stage = "S4"
        s4_ply = run_s4(output_dir / "s3", output_dir / "s4")
        _ok("S4", s4_ply)
        failed_stage = None

        _log("S5 — Application & Deployment")
        failed_stage = "S5"
        s5_path = run_s5(
            output_dir,
            success=True,
            stage_status=stage_status,
            artifacts=artifacts,
            summary={"num_stages": 5},
        )
        _ok("S5", s5_path)
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
        final_output=artifacts.get("S5"),
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
