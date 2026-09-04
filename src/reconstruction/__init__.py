"""S3 — 3D Reconstruction (sealed).

Public integration surface
--------------------------
- :func:`run_s3` — single entry point invoked by the pipeline
  orchestrator.
- :class:`S3Contract` — canonical Pydantic output of S3.

Reusable components exposed for the pipeline orchestrator (cross-
subsystem composition lives there):
- :class:`CameraCalibration` — generic camera calibration record.
- :class:`OpenCVCameraCalibrationLoader` — parses OpenCV YAML files.

Everything else (engines, geometry, input adapters, models, output
packager, preprocessing, quality evaluator, the S2→S3 bridge) lives
under ``reconstruction._internal``.
"""

from ._internal.calibration.loader import OpenCVCameraCalibrationLoader
from ._internal.contracts import S3Contract
from ._internal.models.calibration import CameraCalibration, CameraCalibrationError
from .interface import run_s3

# Expose internal modules for testing
from . import _internal

__all__ = [
    "run_s3",
    "S3Contract",
    "CameraCalibration",
    "CameraCalibrationError",
    "OpenCVCameraCalibrationLoader",
    "_internal",
]
