"""S1 — Visual Perception (sealed).

Public integration surface
--------------------------
- :func:`run_s1` — the only function the pipeline orchestrator is
  allowed to call to run the subsystem.
- :class:`S1Output` — the in-memory return type of :func:`run_s1`,
  exposed because the orchestrator needs it to construct the
  ``S1Contract`` boundary payload.
- :class:`S1Config` — the optional configuration dataclass accepted by
  :func:`run_s1`.
- :class:`S1Contract` / :func:`s1_output_to_contract` — the
  Pydantic-boundary types for the S1 → S2 contract.

Anything else in S1 lives under ``src.visual_perception._internal`` and
is **not** importable from outside the package.
"""

from ._internal.config import S1Config
from ._internal.contracts import S1Contract, s1_output_to_contract
from ._internal.types import S1Output
from .interface import run_s1

__all__ = [
    "run_s1",
    "S1Output",
    "S1Config",
    "S1Contract",
    "s1_output_to_contract",
]
