"""S4 — Georeferencing & Validation: contracts.

Defines the Pydantic ``S4Contract`` that constitutes the **S4 → S5
boundary** as documented in
``docs/architecture/contracts/georeferencing-application.md``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class S4Contract(BaseModel):
    """Canonical S4 → S5 boundary payload.

    Carries the georeferenced scene summary, validation metrics,
    coordinate reference, quality status, and known limitations.
    """

    schema_version: str = "1.0.0"

    georeferenced_scene: Dict[str, Any] = Field(default_factory=dict)
    validation_metrics: Dict[str, Any] = Field(default_factory=dict)
    coordinate_reference: Dict[str, Any] = Field(default_factory=dict)
    quality_status: Dict[str, Any] = Field(default_factory=dict)
    known_limitations: List[str] = Field(default_factory=list)

    artifact_paths: Dict[str, str] = Field(default_factory=dict)
    status: str = "completed"
    metadata: Dict[str, Any] = Field(default_factory=dict)


__all__ = ["S4Contract"]
