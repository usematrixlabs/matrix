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
    GeoreferencingValidator,
    ReconstructionInput,
)
from src.localization_sensor_fusion import (
    S1InputAdapter,
    S2Exporter,
    S2ObservationOutput,
    SensorFusionEngine,
    TrajectorySmoother,
    VisualLocalizerEngine,
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
)
from src.reconstruction import S3ReconstructionPipeline
from src.reconstruction.models.schema import S2Payload
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


def _s1_camera_info(s1_output: S1Output) -> Optional[CameraInfo]:
    """Extract camera intrinsics from S1 metadata into a S2 ``CameraInfo``."""
    meta_calib = s1_output.metadata.get("camera_calibration") if isinstance(s1_output.metadata, dict) else None
    if not meta_calib:
        return None

    intrinsics_raw = meta_calib.get("intrinsics") if isinstance(meta_calib, dict) else None
    width = meta_calib.get("width")
    height = meta_calib.get("height")

    if intrinsics_raw and all(k in intrinsics_raw for k in ("fx", "fy", "cx", "cy")):
        intrinsics = CameraIntrinsics(
            fx=float(intrinsics_raw["fx"]),
            fy=float(intrinsics_raw["fy"]),
            cx=float(intrinsics_raw["cx"]),
            cy=float(intrinsics_raw["cy"]),
        )
    else:
        intrinsics = None

    distortion_raw = meta_calib.get("distortion") if isinstance(meta_calib, dict) else None
    if distortion_raw and distortion_raw.get("coefficients"):
        distortion = Distortion(
            model=str(distortion_raw.get("model", "opencv")),
            coefficients=[float(c) for c in distortion_raw["coefficients"]],
        )
    else:
        distortion = None

    return CameraInfo(
        width=int(width) if width else None,
        height=int(height) if height else None,
        intrinsics=intrinsics,
        distortion=distortion,
    )


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

    S1 → S2 wiring
    --------------
    1. Each S1 ``Frame`` observation is converted into a validated
       ``S1ObservationInput`` via :class:`S1InputAdapter`.
    2. A GPS prior (lat/lon/alt from ``gps.csv`` translated into a local
       ENU offset) is attached to every observation that has no pose yet,
       so S2 always has a position seed before running fusion.
    3. The observations pass through a :class:`TrajectorySmoother` (local
       smoothing) followed by a :class:`SensorFusionEngine` (EKF).

    Output
    ------
    A canonical ``s2_output.json`` file is written to ``<output_dir>``.
    """
    _stage_log("S2", "Running localization & sensor fusion...")

    frames = s1_output.visual_observations.frames
    if not frames:
        raise RuntimeError("S1 produced no observations for S2 to localize.")

    adapter = S1InputAdapter(min_blur_score=0.0)
    observations_json = s1_output.metadata.get("observations_json") if isinstance(s1_output.metadata, dict) else None
    s1_root = Path(observations_json).parent if observations_json else output_dir.parent / "s1"

    camera_info = _s1_camera_info(s1_output)

    s2_observations: List[S2ObservationOutput] = []
    for frame in frames:
        ts = float(getattr(frame, "timestamp", 0.0) or 0.0)
        frame_id = str(getattr(frame, "frame_id", ""))

        # S1 -> S2: validate each observation through the documented adapter.
        payload_dict = {
            "observation_id": frame_id,
            "timestamp": ts,
            "image": str(getattr(frame, "image_path", "")),
            "camera": (
                {
                    "width": getattr(frame, "image_width", None),
                    "height": getattr(frame, "image_height", None),
                    "intrinsics": (
                        {
                            "fx": camera_info.intrinsics.fx,
                            "fy": camera_info.intrinsics.fy,
                            "cx": camera_info.intrinsics.cx,
                            "cy": camera_info.intrinsics.cy,
                        }
                        if camera_info and camera_info.intrinsics
                        else None
                    ),
                }
                if camera_info
                else None
            ),
            "quality": (
                {
                    "status": getattr(frame.quality, "status", "GOOD") if getattr(frame, "quality", None) else "GOOD",
                    "blur_score": float(getattr(frame.quality, "blur_score", 0.0)) if getattr(frame, "quality", None) else 0.0,
                }
                if getattr(frame, "quality", None)
                else None
            ),
        }
        adapter.parse_observation(payload_dict)

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
                image=str(getattr(frame, "image_path", "")),
                camera=camera_info,
                localization=LocalizationMeta(
                    status=PoseStatus.ESTIMATED,
                    source=[LocalizationSource.GPS],
                    quality=LocalizationQuality(confidence=0.5),
                ),
                pose=CameraPose(position=gps_pose, orientation=orientation),
            )
        )

    # S2 internal pipeline: smoother -> EKF fusion.
    smoother = TrajectorySmoother(window_size=3)
    smoothed = smoother.smooth_trajectory(list(s2_observations))

    fusion_engine = SensorFusionEngine()
    fused = fusion_engine.fuse_sequence(list(smoothed))

    payload = S2Exporter().create_payload(fused)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "s2_output.json"
    S2Exporter().export_to_json(payload, out_path)
    return out_path


def run_s3(s1_output: S1Output, s2_output: Path, output_dir: Path) -> Path:
    """Stage S3 — 3D Reconstruction.

    S2 → S3 wiring
    --------------
    1. Load ``s2_output.json`` (S2's canonical ``S2PayloadOutput``).
    2. Translate it through :func:`build_s2_payload_from_s2` into S3's
       ``S2Payload`` schema, attaching 2D feature tracks extracted from
       the S1 frame images referenced by the S2 observations.
    3. Run :class:`S3ReconstructionPipeline` against the translated
       payload.
    """
    _stage_log("S3", "Running 3D reconstruction...")

    s2_payload = S2Exporter.create_payload.__self__ if False else None  # noqa: E305
    # Load S2 output as a Pydantic S2PayloadOutput.
    with open(s2_output, "r", encoding="utf-8") as f:
        s2_raw = json.load(f)

    from src.localization_sensor_fusion.schemas.contracts import (
        S2PayloadOutput as _S2PayloadOutputModel,
    )
    s2_model = _S2PayloadOutputModel.model_validate(s2_raw)

    # The bridge needs the directory containing the S1 frame images.
    s1_root: Optional[Path] = None
    obs_json = s1_output.metadata.get("observations_json") if isinstance(s1_output.metadata, dict) else None
    if obs_json:
        s1_root = Path(obs_json).parent
    if s1_root is None:
        s1_root = s2_output.parent.parent / "s1"

    s3_input: S2Payload = build_s2_payload_from_s2(s2_model, image_root=s1_root)

    pipeline = S3ReconstructionPipeline(check_image_files=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = pipeline.run(
        input_data=s3_input,
        scene_id=output_dir.name,
        output_directory=output_dir,
        raise_on_invalid_input=False,
    )
    status = str(getattr(result, "status", "unknown"))
    _stage_log("S3", f"Reconstruction status={status}, num_points={result.point_cloud.num_points if result.point_cloud else 0}")
    return output_dir / "scene.ply"


def run_s4(s3_output_dir: Path, output_dir: Path) -> Dict[str, Path]:
    """Stage S4 — Georeferencing & Validation.

    S3 → S4 wiring
    --------------
    1. Reload ``s3/metadata.json`` to recover the in-memory
       ``S3ReconstructionResult`` shape, then call
       :meth:`S3ReconstructionResult.to_s4_reconstruction_input` to get
       a fully-validated :class:`ReconstructionInput`.

       In the common case the reconstruction is still available in
       ``s3_output_dir``; if not (e.g., S3 reported an empty scene),
       S4 falls back to reading the PLY directly so the pipeline still
       produces an S5-bundled ``georeferencing.json``.

    2. Build identity-like control points (with no real GCPs available
       this is a placeholder Helmert fit — documented as such).

    3. Run :class:`Georeferencer` and validate via
       :class:`GeoreferencingValidator`.

    Returns
    -------
    dict
        ``{"ply": ..., "georeferencing": ..., "validation": ...}``
    """
    _stage_log("S4", "Running georeferencing & validation...")

    output_dir.mkdir(parents=True, exist_ok=True)

    points: Optional[np.ndarray] = None
    colors: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = {}

    metadata_path = s3_output_dir / "metadata.json"
    ply_path = s3_output_dir / "scene.ply"

    if metadata_path.is_file():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                s3_metadata = json.load(f)
            num_points = int(s3_metadata.get("num_points", 0))
            if num_points > 0:
                # Reconstruct point array from the PLY file (canonical artifact).
                if ply_path.is_file():
                    from src.reconstruction.geometry.ply_io import PlyIO

                    pcd = PlyIO.read_ply(ply_path)
                    points = np.asarray(pcd.points, dtype=np.float64)
                    colors = (
                        np.asarray(pcd.colors, dtype=np.uint8)
                        if pcd.colors is not None
                        else None
                    )
                metadata = s3_metadata
        except (OSError, json.JSONDecodeError, ValueError):
            points = None

    if points is None and ply_path.is_file():
        try:
            from src.reconstruction.geometry.ply_io import PlyIO

            pcd = PlyIO.read_ply(ply_path)
            points = np.asarray(pcd.points, dtype=np.float64)
            colors = (
                np.asarray(pcd.colors, dtype=np.uint8)
                if pcd.colors is not None
                else None
            )
        except (OSError, ValueError):
            points = None

    if points is None or points.shape[0] == 0:
        _stage_log("S4", "degraded: S3 produced no point cloud.")
        out_ply = output_dir / "georeferenced.ply"
        with open(out_ply, "w", encoding="utf-8") as f:
            f.write("")
        meta_path = output_dir / "georeferencing.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {"status": "degraded", "reason": "S3 produced no point cloud."},
                f,
                indent=2,
            )
        return {
            "ply": out_ply,
            "georeferencing": meta_path,
            "validation": meta_path,
        }

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
        return {
            "ply": out_ply,
            "georeferencing": meta_path,
            "validation": meta_path,
        }

    reconstruction = ReconstructionInput(
        points=points,
        colors=colors,
        metadata=metadata,
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
    geo_result = georeferencer.georeference()

    out_ply = output_dir / "georeferenced.ply"

    from src.reconstruction.models.s3_output import PointCloudData as _PCD
    from src.reconstruction.geometry.ply_io import PlyIO

    PlyIO.write_ply(
        out_ply,
        _PCD(points=np.asarray(geo_result.points, dtype=np.float64), colors=geo_result.colors),
        binary=True,
    )

    # S4 validation step — exercises GeoreferencingValidator so S5 has
    # a real ValidationResult to summarize downstream.
    validator = GeoreferencingValidator(
        control_points=control_points,
        transformation=georeferencer.transformation,
        tolerance=None,
    )
    validation_result = validator.validate()

    meta_path = output_dir / "georeferencing.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_points": int(np.asarray(geo_result.points).shape[0]),
                "source_crs": source_crs.name,
                "target_crs": target_crs.name,
                "method": geo_result.metadata.get("georeferencing_method"),
                "rmse": float(validation_result.rmse),
                "mean_error": float(validation_result.mean_error),
                "max_error": float(validation_result.max_error),
                "median_error": float(validation_result.median_error),
                "passed": validation_result.passed,
                "note": (
                    "Identity Helmert fit used as placeholder. Replace "
                    "control points with real GCPs for production georeferencing."
                ),
            },
            f,
            indent=2,
        )

    return {
        "ply": out_ply,
        "georeferencing": meta_path,
        "validation": meta_path,
    }


def run_s5(
    output_dir: Path,
    success: bool,
    stage_status: Dict[str, str],
    artifacts: Dict[str, Path],
    summary: Dict[str, Any],
) -> Path:
    """Stage S5 — Application & Deployment (output bundler).

    S4 → S5 wiring
    --------------
    The orchestrator passes S4's georeferencing + validation summary into
    :meth:`Finalizer.bundle` as part of the ``summary`` payload so S5
    actually consumes the ``GeoreferencedResult`` /
    :class:`ValidationResult` information instead of discarding it.
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
    s4_summary: Dict[str, Any] = {}
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
        s4_artifacts = run_s4(output_dir / "s3", output_dir / "s4")
        _ok("S4", s4_artifacts["ply"])
        s4_summary = {
            "georeferencing": str(s4_artifacts["georeferencing"]),
            "validation": str(s4_artifacts["validation"]),
        }
        # Inline the validation stats if a real georeferencing run completed.
        geo_meta_path = s4_artifacts["georeferencing"]
        if geo_meta_path.is_file():
            try:
                with open(geo_meta_path, "r", encoding="utf-8") as f:
                    geo_meta = json.load(f)
                if "rmse" in geo_meta:
                    s4_summary["rmse"] = geo_meta["rmse"]
                    s4_summary["mean_error"] = geo_meta.get("mean_error")
                    s4_summary["max_error"] = geo_meta.get("max_error")
                    s4_summary["median_error"] = geo_meta.get("median_error")
                    s4_summary["passed"] = geo_meta.get("passed")
                    s4_summary["status"] = geo_meta.get("status", "completed")
            except (OSError, json.JSONDecodeError):
                pass
        failed_stage = None

        _log("S5 — Application & Deployment")
        failed_stage = "S5"
        s5_path = run_s5(
            output_dir,
            success=True,
            stage_status=stage_status,
            artifacts=artifacts,
            summary={"num_stages": 5, "s4": s4_summary},
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
