"""Classical ORB + Hamming matcher backend.

This is the historical default visual correspondence backend for S2.
It deliberately depends only on OpenCV (already a Matrix dependency),
so it works everywhere without any learned model.

Behavior matches the previous in-engine behavior of
``VisualLocalizerEngine.extract_and_match_features`` for the
2D-to-2D case; PnP/3D matching is delegated to the engine.
"""
from __future__ import annotations

from typing import Union

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "OpenCV (cv2) is required for the classical matcher backend."
    ) from exc

from .base import FeatureMatcher, MatchResult


PathOrImage = Union[str, np.ndarray]


def _load_bgr(image: PathOrImage) -> np.ndarray:
    if isinstance(image, str):
        arr = cv2.imread(image, cv2.IMREAD_COLOR)
        if arr is None:
            raise FileNotFoundError(f"Cannot read image: {image}")
        return arr
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Unsupported image type: {type(image).__name__}")
    if image.ndim == 2:
        return image
    return image


class ClassicalMatcher:
    """ORB + BFMatcher (Hamming, cross-check) backend.

    Mirrors the original matcher embedded in
    ``VisualLocalizerEngine``. Returns pixel correspondences only --
    3D associations are an engine-level concern.
    """

    backend_name = "classical"

    def __init__(self, n_features: int = 1000) -> None:
        self._orb = cv2.ORB_create(nfeatures=int(n_features))
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def match(self, image0: PathOrImage, image1: PathOrImage) -> MatchResult:
        img0 = _load_bgr(image0)
        img1 = _load_bgr(image1)

        gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY) if img0.ndim == 3 else img0
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim == 3 else img1

        kp0, desc0 = self._orb.detectAndCompute(gray0, None)
        kp1, desc1 = self._orb.detectAndCompute(gray1, None)
        if desc0 is None or desc1 is None or len(desc0) == 0 or len(desc1) == 0:
            return MatchResult(
                points0=np.empty((0, 2), dtype=np.float64),
                points1=np.empty((0, 2), dtype=np.float64),
                scores=None,
            )

        raw = self._bf.match(desc0, desc1)
        if not raw:
            return MatchResult(
                points0=np.empty((0, 2), dtype=np.float64),
                points1=np.empty((0, 2), dtype=np.float64),
                scores=None,
            )

        pts0 = np.array([kp0[m.queryIdx].pt for m in raw], dtype=np.float64)
        pts1 = np.array([kp1[m.trainIdx].pt for m in raw], dtype=np.float64)
        return MatchResult(points0=pts0, points1=pts1, scores=None)


def _assert_protocol(m: ClassicalMatcher) -> None:
    assert isinstance(m, FeatureMatcher) or hasattr(m, "match")


__all__ = ["ClassicalMatcher"]
