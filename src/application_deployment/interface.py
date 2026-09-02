"""S5 — Application & Deployment: single integration entry point.

Public API
----------
- :func:`run_s5` — the only function the pipeline orchestrator is
  allowed to call. Accepts the canonical ``S4Contract``, the per-run
  status / artifacts / summary metadata, and produces an ``S5Contract``
  whose contents are written to ``<output_dir>/s5/final_output.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from ._internal.contracts import S5Contract
from ._internal.finalizer import Finalizer


def run_s5(
    s4_contract: Any,
    output_dir: Path,
    success: bool,
    stage_status: Dict[str, str],
    artifacts: Optional[Dict[str, str]] = None,
    summary: Optional[Dict[str, Any]] = None,
    config: Optional[dict] = None,
) -> S5Contract:
    """Single integration entry point for S5.

    Parameters
    ----------
    s4_contract
        Canonical S4 wire-format payload (``S4Contract``). Typed as
        ``Any`` at runtime because S5 must not import S4's types; the
        expected shape is documented in
        ``docs/architecture/contracts/georeferencing-application.md``.
    output_dir : Path
        Per-run output directory; S5 writes its bundle into
        ``<output_dir>/s5/``.
    success : bool
        Whether the pipeline reached S5 without an earlier hard fail.
    stage_status : dict
        ``{"S1": "completed", ...}`` per-stage status map.
    artifacts : dict, optional
        Mapping of ``stage -> artifact_path`` for downstream display.
    summary : dict, optional
        Aggregated metrics (RMSE, num_points, etc.).
    config : dict, optional
        Reserved for future tuning. Ignored for now.

    Returns
    -------
    S5Contract
        Validated Pydantic S5 output.
    """
    finalizer = Finalizer(output_dir=output_dir / "s5")
    final = finalizer.bundle(
        scene_id=Path(output_dir).name,
        success=success,
        stage_status=stage_status,
        artifacts=artifacts,
        summary=summary,
    )

    known_limitations = list(getattr(s4_contract, "known_limitations", []) or [])
    quality_status = dict(getattr(s4_contract, "quality_status", {}) or {})

    return S5Contract(
        manifest={
            "schema_version": Finalizer.SCHEMA_VERSION,
            "scene_id": final.scene_id,
            "output_dir": final.output_dir,
            "success": final.success,
            "stage_status": dict(final.stage_status),
            "created_at": final.created_at,
            "status": (
                "complete"
                if success
                else "failed"
            ),
        },
        artifacts=dict(final.artifacts),
        summary=dict(final.summary),
        known_limitations=known_limitations,
        metadata={
            "quality_status": quality_status,
        },
    )


__all__ = ["run_s5"]
