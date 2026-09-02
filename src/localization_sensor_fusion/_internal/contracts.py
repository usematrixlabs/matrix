"""S2 — Localization & Sensor Fusion: contracts.

Defines the Pydantic ``S2Contract`` that constitutes the **S2 → S3
boundary** as documented in
``docs/architecture/contracts/localization-reconstruction.md``.

The ``S2Contract`` mirrors the JSON schema written to
``<output_dir>/s2/s2_output.json``; the orchestrator hands it forward
to S3's :func:`run_s3`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .schemas.contracts import (
    S2ObservationOutput,
    S2PayloadOutput,
)


class S2Contract(S2PayloadOutput):
    """Canonical P2 output of S2.

    Concretely the same shape as ``S2PayloadOutput``; we re-export under
    a contract-flavored name so the orchestrator and S3 import a
    producer-owned boundary type. New metadata fields may be added as
    the S2→S3 contract evolves without changing the wire format.
    """

    schema_version: str = "0.1"
    coordinate_frame: str = "local"
    observations: List[S2ObservationOutput] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


def contract_to_observations(contract: S2Contract) -> List[S2ObservationOutput]:
    """Return the observation list carried by ``S2Contract``."""
    return list(contract.observations)


__all__ = ["S2Contract", "contract_to_observations"]
