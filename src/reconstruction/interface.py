"""S3 — 3D Reconstruction: single integration entry point.

Public API
----------
- :func:`run_s3` — the only function the pipeline orchestrator is
  allowed to call. Accepts the canonical ``S2Contract``, an image-root
  directory, produces an ``S3Contract``, and writes ``scene.ply`` +
  ``metadata.json`` to ``output_dir``.

The internal S2→S3 bridge (which adds per-frame 2D feature tracks so
that S3 can triangulate) lives under ``_internal`` and is invoked by
``run_s3`` — S3 never reaches into S2's internals; the upstream wire
format (``S2Contract``) is consumed as a duck-typed object.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ._internal.calibration.loader import OpenCVCameraCalibrationLoader
from ._internal.contracts import S3Contract, s3_result_to_contract
from ._internal.models.calibration import CameraCalibration
from ._internal.pipeline import S3ReconstructionPipeline
from ._internal.s2_to_s3_bridge import build_s2_payload_from_contract


def run_s3(
    s2_contract: Any,
    image_root: Path,
    output_dir: Path,
    config: Optional[dict] = None,
    calibration: Optional[CameraCalibration] = None,
) -> S3Contract:
    """Single integration entry point for S3.

    Parameters
    ----------
    s2_contract
        Canonical S2 wire-format payload (``S2Contract``). Typed as
        ``Any`` at runtime because S3 must not import S2's types; the
        expected shape is documented in
        ``docs/architecture/contracts/localization-reconstruction.md``
        and in :mod:`src.reconstruction._internal.s2_to_s3_bridge`.
    image_root : Path
        Directory where the S1 frame images referenced by S2 live.
    output_dir : Path
        Directory where ``scene.ply`` and ``metadata.json`` are written.
    config : dict, optional
        Reserved for future tuning. May contain a ``"calibration_path"``
        key pointing to an OpenCV YAML calibration file. Explicitly
        passing ``calibration`` overrides this.
    calibration : CameraCalibration, optional
        Pre-loaded camera calibration. When ``None`` and ``config`` does
        not contain ``"calibration_path"``, S3 falls back to the existing
        heuristic intrinsics derived from image dimensions (documented
        in ``docs/architecture/system-architecture.md`` §8).

    Returns
    -------
    S3Contract
        Validated Pydantic S3 output. The orchestrator hands this to S4.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_root = Path(image_root)

    # Resolve calibration from explicit argument or config dict.
    resolved_calibration: Optional[CameraCalibration] = calibration
    if resolved_calibration is None and isinstance(config, dict):
        calib_path = config.get("calibration_path")
        if calib_path:
            resolved_calibration = OpenCVCameraCalibrationLoader.load_from_file(calib_path)

    s2_payload = build_s2_payload_from_contract(
        s2_contract,
        image_root=image_root,
    )

    pipeline = S3ReconstructionPipeline(
        check_image_files=False,
        calibration=resolved_calibration,
    )
    result = pipeline.run(
        input_data=s2_payload,
        scene_id=output_dir.name,
        output_directory=output_dir,
        raise_on_invalid_input=False,
    )

    artifact_paths = {
        "ply": str(output_dir / "scene.ply"),
        "metadata": str(output_dir / "metadata.json"),
    }

    return s3_result_to_contract(result, artifact_paths=artifact_paths)


__all__ = ["run_s3"]
