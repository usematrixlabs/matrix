"""S5 — Application & Deployment: contracts.

Defines the Pydantic ``S5Contract`` produced by :func:`run_s5`.

The contract carries the pipeline ``manifest`` (top-level status,
stage_status, success flag), the per-subsystem ``artifacts`` (paths to
the canonical outputs S1–S4 produced), and a ``summary`` of the
end-to-end metrics the downstream application / API will display.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class S5Contract(BaseModel):
    """Canonical S5 (Application & Deployment) output."""

    schema_version: str = "1.0.0"

    manifest: Dict[str, Any] = Field(default_factory=dict)
    artifacts: Dict[str, str] = Field(default_factory=dict)
    summary: Dict[str, Any] = Field(default_factory=dict)

    known_limitations: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


__all__ = ["S5Contract"]
