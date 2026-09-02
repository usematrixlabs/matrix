"""S5 — Application & Deployment (sealed).

Public integration surface
--------------------------
- :func:`run_s5` — single entry point invoked by the pipeline
  orchestrator.
- :class:`S5Contract` — canonical Pydantic output of S5.

Everything else (finalizer, dead-code alternates) lives under
``src.application_deployment._internal`` (or has been removed — see
Phase G of the isolation refactor).
"""

from ._internal.contracts import S5Contract
from .interface import run_s5

__all__ = ["run_s5", "S5Contract"]
