"""S2 — Localization & Sensor Fusion: single integration entry point.

Public API
----------
- :func:`run_s2` — the only function the pipeline orchestrator is
  allowed to call. Accepts the canonical ``S1Contract`` and a GPS CSV,
  produces an ``S2Contract``, and writes the canonical
  ``s2_output.json`` to ``output_dir``.

Everything else (adapters, engines, fusion, exporters) lives under
``src.localization_sensor_fusion._internal``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from ._internal.contracts import S2Contract
from ._internal.engines.trajectory_smoother import TrajectorySmoother
from ._internal.exporters.s2_exporter import S2Exporter
from ._internal.fusion.fusion_engine import SensorFusionEngine
from ._internal.schemas.contracts import (
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
)


@dataclass
class _GpsAnchor:
    lat0: float
    lon0: float
    alt0: float


def _read_gps_anchor(gps_path: Path) -> Optional[_GpsAnchor]:
    """Return the first GPS row as a local ENU anchor."""
    try:
        with open(gps_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                return _GpsAnchor(
                    lat0=float(row.get("drone_lat") or row.get("vehicle_lat") or 0.0),
                    lon0=float(row.get("drone_lon") or row.get("vehicle_lon") or 0.0),
                    alt0=float(row.get("drone_altitude_m") or row.get("vehicle_altitude_m") or 0.0),
                )
    except (OSError, ValueError, KeyError):
        return None
    return None


def _gps_position_at(timestamp: float, gps_path: Path, anchor: _GpsAnchor) -> Position:
    """Return a local-frame Position interpolated from gps.csv at ``timestamp``."""
    best_dt = float("inf")
    best_row: Optional[dict] = None
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
        best_row = None

    if best_row is None:
        return Position(x=0.0, y=0.0, z=0.0)

    lat = float(best_row.get("drone_lat") or best_row.get("vehicle_lat") or anchor.lat0)
    lon = float(best_row.get("drone_lon") or best_row.get("vehicle_lon") or anchor.lon0)
    alt = float(best_row.get("drone_altitude_m") or best_row.get("vehicle_altitude_m") or anchor.alt0)

    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * max(np.cos(np.deg2rad(anchor.lat0)), 1e-6)

    east = (lon - anchor.lon0) * meters_per_deg_lon
    north = (lat - anchor.lat0) * meters_per_deg_lat
    up = alt - anchor.alt0

    return Position(x=float(east), y=float(north), z=float(up))


def _camera_info_from_s1(s1_contract: Any) -> Optional[CameraInfo]:
    """Extract S2 CameraInfo from the S1 contract's camera calibration metadata."""
    cam = s1_contract.camera
    if not cam:
        return None

    intrinsics_raw = cam.get("intrinsics") if isinstance(cam, dict) else None
    if intrinsics_raw and all(k in intrinsics_raw for k in ("fx", "fy", "cx", "cy")):
        intrinsics = CameraIntrinsics(
            fx=float(intrinsics_raw["fx"]),
            fy=float(intrinsics_raw["fy"]),
            cx=float(intrinsics_raw["cx"]),
            cy=float(intrinsics_raw["cy"]),
        )
    else:
        intrinsics = None

    distortion_raw = cam.get("distortion") if isinstance(cam, dict) else None
    if distortion_raw and distortion_raw.get("coefficients"):
        distortion = Distortion(
            model=str(distortion_raw.get("model", "opencv")),
            coefficients=[float(c) for c in distortion_raw["coefficients"]],
        )
    else:
        distortion = None

    width = cam.get("width") if isinstance(cam, dict) else None
    height = cam.get("height") if isinstance(cam, dict) else None
    return CameraInfo(
        width=int(width) if width else None,
        height=int(height) if height else None,
        intrinsics=intrinsics,
        distortion=distortion,
    )


def run_s2(
    s1_contract: Any,
    gps_path: Path,
    output_dir: Path,
    config: Optional[dict] = None,
) -> S2Contract:
    """Single integration entry point for S2.

    Parameters
    ----------
    s1_contract : S1Contract (duck-typed)
        Canonical wire-format payload from S1. Typed as ``Any`` at
        runtime because S2 must not import S1's types; the expected
        shape is documented in
        ``docs/architecture/contracts/perception-localization.md``.
    gps_path : Path
        Path to a GPS CSV (any reasonable schema with ``drone_lat`` /
        ``drone_lon`` / ``drone_altitude_m`` / ``video_time_s`` columns).
    output_dir : Path
        Directory where the canonical ``s2_output.json`` is written.
    config : dict, optional
        Reserved for future tuning (window size, fusion params). Ignored
        for now.

    Returns
    -------
    S2Contract
        Validated Pydantic S2 output. The orchestrator may either pass
        this to S3 directly or serialize it to JSON via
        :meth:`S2Contract.model_dump_json`.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not s1_contract.observations:
        raise RuntimeError("S1 produced no observations for S2 to localize.")

    camera_info = _camera_info_from_s1(s1_contract)

    anchor = _read_gps_anchor(Path(gps_path)) if Path(gps_path).exists() else None

    observations: List[S2ObservationOutput] = []
    for obs in s1_contract.observations:
        ts = float(obs.get("timestamp", 0.0) or 0.0)
        observation_id = str(obs.get("observation_id", ""))

        if anchor is not None and Path(gps_path).exists():
            gps_pose = _gps_position_at(ts, Path(gps_path), anchor)
        else:
            gps_pose = Position(x=0.0, y=0.0, z=0.0)

        orientation = QuaternionOrientation(qx=0.0, qy=0.0, qz=0.0, qw=1.0)
        observations.append(
            S2ObservationOutput(
                observation_id=observation_id,
                timestamp=ts,
                image=str(obs.get("image", "")),
                camera=camera_info,
                localization=LocalizationMeta(
                    status=PoseStatus.ESTIMATED,
                    source=[LocalizationSource.GPS],
                    quality=LocalizationQuality(confidence=0.5),
                ),
                pose=CameraPose(position=gps_pose, orientation=orientation),
            )
        )

    smoother = TrajectorySmoother(window_size=3)
    smoothed = smoother.smooth_trajectory(list(observations))

    fusion_engine = SensorFusionEngine()
    fused = fusion_engine.fuse_sequence(list(smoothed))

    payload = S2Exporter().create_payload(fused)

    s2_contract = S2Contract(
        schema_version=payload.schema_version,
        coordinate_frame=payload.coordinate_frame,
        units=payload.units,
        observations=payload.observations,
        metadata={
            "source": "S2 Localization & Sensor Fusion",
            "num_observations": len(payload.observations),
            "gps_anchor": (
                {
                    "lat0": anchor.lat0,
                    "lon0": anchor.lon0,
                    "alt0": anchor.alt0,
                }
                if anchor is not None
                else None
            ),
        },
    )

    out_path = output_dir / "s2_output.json"
    S2Exporter().export_to_json(s2_contract, out_path)
    return s2_contract


__all__ = ["run_s2"]
