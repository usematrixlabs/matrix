"""S3 — 3D Reconstruction (sealed).

Public integration surface
--------------------------
- :func:`run_s3` — single entry point invoked by the pipeline
  orchestrator.
- :class:`S3Contract` — canonical Pydantic output of S3.

Everything else (engines, geometry, input adapters, models, output
packager, preprocessing, quality evaluator, the S2→S3 bridge) lives
under ``src.reconstruction._internal``.
"""

from ._internal.contracts import S3Contract
from .interface import run_s3

__all__ = ["run_s3", "S3Contract"]
