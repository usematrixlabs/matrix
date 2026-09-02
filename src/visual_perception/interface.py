"""S1 — Visual Perception: single integration entry point.

Public API
----------
- :func:`run_s1` — the only function the pipeline orchestrator is allowed
  to call. It runs the full Visual Perception subsystem and returns the
  canonical :class:`S1Output`.

Everything else in S1 is private to ``src.visual_perception._internal``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ._internal.config import S1Config
from ._internal.pipeline import S1Pipeline
from ._internal.types import S1Output


def run_s1(
    video_path: Path,
    output_dir: Path,
    config: Optional[S1Config] = None,
) -> S1Output:
    """Single integration entry point for S1 (Visual Perception).

    Parameters
    ----------
    video_path
        Path to the input UAV video file.
    output_dir
        Directory where S1 writes ``observations.json`` and ``frames/``.
    config
        Optional pre-built :class:`S1Config`. When omitted, a sensible
        default is constructed (fixed-interval sampling, WARNING log
        level) that matches the orchestrator's long-standing behavior.

    Returns
    -------
    S1Output
        Canonical S1 result. The orchestrator is expected to convert this
        into the S2 input contract; S1 itself never reaches across the
        boundary.
    """
    cfg = config or S1Config(
        video_path=str(video_path),
        output_dir=str(output_dir),
        frames_dir=str(output_dir / "frames"),
        keyframes_dir=str(output_dir / "keyframes"),
        log_level="WARNING",
        sampling_mode="fixed",
        sampling_interval=60,
    )
    return S1Pipeline(config=cfg).run()


__all__ = ["run_s1"]
