"""S4 — Georeferencing & Validation (sealed).

Public integration surface
--------------------------
- :func:`run_s4` — single entry point invoked by the pipeline
  orchestrator.
- :class:`S4Contract` — canonical Pydantic output of S4.

Everything else (control points, CRS, Helmert, input validator,
georeferencer, IO helpers) lives under
``src.georeferencing_validation._internal``.
"""

from ._internal.contracts import S4Contract
from .interface import run_s4

__all__ = ["run_s4", "S4Contract"]
