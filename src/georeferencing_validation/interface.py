"""S4 — Georeferencing & Validation: single integration entry point.

Public API
----------
- :func:`run_s4` — the only function the pipeline orchestrator is
  allowed to call. Accepts the canonical ``S3Contract``, runs
  georeferencing + validation, produces an ``S4Contract``, and writes
  ``georeferenced.ply`` + ``georeferencing.json`` to ``output_dir``.

The placeholder Helmert fit with identity-like control points lives
here; replacing it with real GCPs is a configuration / data concern
that flows through ``config`` and never reaches into S3.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ._internal.contracts import S4Contract
from ._internal.control_points import ControlPoints
from ._internal.crs import CoordinateReference
from ._internal.georeferencer import Georeferencer, GeoreferencedResult
from ._internal.geometry_io import (
    load_point_cloud_from_s3_artifacts,
    write_georeferenced_ply,
)
from ._internal.input import ReconstructionInput
from ._internal.validator import GeoreferencingValidator


def _empty_s4_contract(
    output_dir: Path,
    reason: str,
    num_points: int = 0,
    metadata: Optional[Dict[str, Any]] = None,
) -> S4Contract:
    """Build a degraded ``S4Contract`` for the empty-point-cloud case."""
    out_ply = output_dir / "georeferenced.ply"
    out_ply.write_text("", encoding="utf-8")
    meta_path = output_dir / "georeferencing.json"
    meta_path.write_text(
        json.dumps(
            {
                "status": "degraded",
                "reason": reason,
                "num_points": int(num_points),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return S4Contract(
        status="degraded",
        artifact_paths={
            "ply": str(out_ply),
            "georeferencing": str(meta_path),
            "validation": str(meta_path),
        },
        metadata={"reason": reason, **(metadata or {})},
    )


def run_s4(
    s3_contract: Any,
    output_dir: Path,
    config: Optional[dict] = None,
) -> S4Contract:
    """Single integration entry point for S4.

    Parameters
    ----------
    s3_contract
        Canonical S3 wire-format payload (``S3Contract``). Typed as
        ``Any`` at runtime because S4 must not import S3's types; the
        expected shape is documented in
        ``docs/architecture/contracts/reconstruction-georeferencing.md``.
    output_dir : Path
        Directory where ``georeferenced.ply`` and ``georeferencing.json``
        are written.
    config : dict, optional
        Reserved for future tuning (e.g., real GCP arrays, target CRS).
        Ignored for now.

    Returns
    -------
    S4Contract
        Validated Pydantic S4 output. The orchestrator hands this to S5.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    points, colors, metadata = load_point_cloud_from_s3_artifacts(s3_contract)

    if points is None or points.shape[0] == 0:
        return _empty_s4_contract(
            output_dir,
            reason="S3 produced no point cloud.",
        )

    if points.shape[0] < 3:
        return _empty_s4_contract(
            output_dir,
            reason=(
                f"S3 produced only {int(points.shape[0])} points; "
                "cannot fit a Helmert transformation."
            ),
            num_points=int(points.shape[0]),
        )

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
    target_crs = CoordinateReference(
        name="LOCAL_GEOGRAPHIC_PLACEHOLDER", units="meters", dimension=3
    )

    georeferencer = Georeferencer(
        reconstruction_data=reconstruction,
        control_points=control_points,
        source_crs=source_crs,
        target_crs=target_crs,
    )
    geo_result: GeoreferencedResult = georeferencer.georeference()

    out_ply = output_dir / "georeferenced.ply"
    write_georeferenced_ply(out_ply, geo_result)

    validator = GeoreferencingValidator(
        control_points=control_points,
        transformation=georeferencer.transformation,
        tolerance=None,
    )
    validation_result = validator.validate()

    meta_path = output_dir / "georeferencing.json"
    contract_payload = geo_result.export_contract_payload()
    meta_payload = {
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
        "contract": contract_payload,
    }
    meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")

    return S4Contract(
        status="completed",
        georeferenced_scene=contract_payload.get("geo_referenced_scene", {}),
        validation_metrics=contract_payload.get("validation_metrics", {}),
        coordinate_reference=contract_payload.get("coordinate_reference", {}),
        quality_status=contract_payload.get("quality_status", {}),
        known_limitations=list(contract_payload.get("known_limitations", [])),
        artifact_paths={
            "ply": str(out_ply),
            "georeferencing": str(meta_path),
            "validation": str(meta_path),
        },
        metadata={
            "num_points": int(np.asarray(geo_result.points).shape[0]),
            "rmse": float(validation_result.rmse),
            "passed": bool(validation_result.passed),
            "method": geo_result.metadata.get("georeferencing_method"),
        },
    )


__all__ = ["run_s4"]
