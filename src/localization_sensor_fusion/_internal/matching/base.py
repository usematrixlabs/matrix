"""S2 visual-correspondence backend interface.

The :class:`FeatureMatcher` protocol and the :class:`MatchResult` data
class together form the S2 internal contract between feature matching
and downstream pose estimation. Backends (classical ORB, LightGlue, ...)
implement :class:`FeatureMatcher` and never leak their native types
outside this module.

This boundary is internal to S2; the S1->S2 and S2->S3 wire contracts
are unaffected by which backend is active.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Tuple, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class MatchResult:
    """Pixel correspondences between two images.

    Attributes
    ----------
    points0 : (N, 2) float64 array
        Keypoints in image 0, in image-0 pixel coordinates.
    points1 : (N, 2) float64 array
        Keypoints in image 1, in image-1 pixel coordinates.
        ``points0.shape == points1.shape`` is always true.
    scores : (N,) float32 array or None
        Per-match confidence. ``None`` if the backend does not expose
        scores (the classical backend currently does not).
    """

    points0: np.ndarray
    points1: np.ndarray
    scores: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.points0.shape != self.points1.shape:
            raise ValueError(
                f"MatchResult shape mismatch: points0={self.points0.shape} "
                f"points1={self.points1.shape}"
            )
        if self.points0.ndim != 2 or self.points0.shape[1] != 2:
            raise ValueError(
                f"MatchResult points must be (N,2); got {self.points0.shape}"
            )

    @property
    def n_matches(self) -> int:
        return int(self.points0.shape[0])

    def is_empty(self) -> bool:
        return self.n_matches == 0


@runtime_checkable
class FeatureMatcher(Protocol):
    """Pluggable visual-correspondence backend.

    Implementations must accept either filesystem paths or in-memory BGR
    uint8 images and return a :class:`MatchResult`. They must not raise
    on bad input -- an empty :class:`MatchResult` is the contract for
    "no usable correspondences".
    """

    backend_name: str

    def match(
        self,
        image0: str | np.ndarray,
        image1: str | np.ndarray,
    ) -> MatchResult: ...


def build_matcher(config: Optional[dict] = None) -> FeatureMatcher:
    """Construct a matcher from a configuration mapping.

    Expected keys (all optional):

    * ``backend``: ``"classical"`` (default) or ``"lightglue"``.
    * ``max_num_keypoints``: int, passed to the LightGlue backend
      (ignored by classical).

    Unknown backends raise ``ValueError`` so misconfiguration is loud.
    """
    cfg = dict(config or {})
    backend = str(cfg.get("backend", "classical")).lower()

    if backend == "classical":
        from .classical import ClassicalMatcher

        return ClassicalMatcher()

    if backend == "lightglue":
        from .lightglue import LightGlueMatcher

        max_kp = int(cfg.get("max_num_keypoints", 2048))
        return LightGlueMatcher(max_num_keypoints=max_kp)

    raise ValueError(
        f"Unknown matcher backend: {backend!r}. "
        "Expected 'classical' or 'lightglue'."
    )


__all__ = [
    "FeatureMatcher",
    "MatchResult",
    "build_matcher",
]
