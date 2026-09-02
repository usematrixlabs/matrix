"""S2 — Localization & Sensor Fusion (sealed).

Public integration surface
--------------------------
- :func:`run_s2` — single entry point invoked by the pipeline
  orchestrator.
- :class:`S2Contract` — canonical Pydantic output of S2 (matches
  ``s2_output.json`` shape).

Everything else (adapters, engines, fusion, exporters, schemas) lives
under ``src.localization_sensor_fusion._internal``.
"""

from ._internal.contracts import S2Contract
from .interface import run_s2

__all__ = ["run_s2", "S2Contract"]
