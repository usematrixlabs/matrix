"""S1 — Visual Perception: contract types.

Defines the Pydantic ``S1Contract`` (and friends) that constitute the
**S1 → S2 boundary** as documented in
``docs/architecture/contracts/perception-localization.md``.

These types live under ``_internal`` because they are producer-owned
boundary types. They are exposed via ``src.visual_perception`` only as
the canonical ``S1Output`` dataclass (a thin shell over the same data).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .types import S1Output


class S1Contract(BaseModel):
    """Pydantic shape of the S1 → S2 boundary payload.

    Mirrors ``S1Output.to_dict()``; carries both the observation list and
    the per-observation calibration/quality metadata S2 needs to begin
    localization.
    """

    schema_version: str = "1.2.0"
    subsystem: str = "S1_Visual_Perception"
    video_source: str = ""
    created_at: str = ""

    total_observations: int = 0
    keyframe_count: int = 0
    keyframe_density: float = 0.0

    status: str = "initialized"
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)

    camera: Optional[Dict[str, Any]] = None
    observations: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


def s1_output_to_contract(s1_output: S1Output) -> S1Contract:
    """Convert an :class:`S1Output` (dataclass) into an :class:`S1Contract`.

    The orchestrator is expected to call this exactly once on the
    boundary; downstream subsystems (S2 and beyond) operate on the
    validated ``S1Contract`` and never touch ``S1Output`` directly.
    """
    payload = s1_output.to_dict()
    observations = payload.get("visual_observations", {}).get("frames", [])
    metadata = payload.get("metadata", {}) or {}
    camera = metadata.get("camera_calibration")
    keyframes = payload.get("visual_observations", {}).get("keyframes", []) or []

    return S1Contract(
        schema_version="1.2.0",
        subsystem="S1_Visual_Perception",
        video_source=str(metadata.get("video_source", "")),
        created_at=str(metadata.get("created_at", "")),
        total_observations=len(observations),
        keyframe_count=len(keyframes),
        keyframe_density=(len(keyframes) / len(observations)) if observations else 0.0,
        status=payload.get("status", "initialized"),
        warnings=list(payload.get("warnings", [])),
        errors=list(payload.get("errors", [])),
        camera=camera,
        observations=observations,
        metadata=metadata,
    )


__all__ = ["S1Contract", "s1_output_to_contract"]
