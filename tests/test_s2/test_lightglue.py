"""Smoke tests for the LightGlue matcher backend.

These tests do NOT depend on real drone footage. They synthesize a pair
of overlapping images so we can verify:

1. The vendored LightGlue installs and imports correctly.
2. The :class:`FeatureMatcher` adapter contract is honoured.
3. Match coordinates are finite and shape-consistent.

A real-frame integration test belongs in the S2 pipeline tests, not here.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from localization_sensor_fusion._internal.matching import (
    FeatureMatcher,
    MatchResult,
    build_matcher,
)


def _synth_pair(tmp_path: Path, seed: int = 0) -> tuple[str, str]:
    """Write two overlapping synthetic frames to disk and return paths."""
    import cv2

    rng = np.random.default_rng(seed)
    img = (rng.random((360, 480, 3)) * 255).astype(np.uint8)
    for _ in range(300):
        c = (int(rng.integers(0, 480)), int(rng.integers(0, 360)))
        r = int(rng.integers(2, 8))
        cv2.circle(img, c, r, (int(rng.integers(50, 255)),) * 3, -1)

    M = np.array([[1.0, 0.0, 30.0], [0.0, 1.0, 12.0]], dtype=np.float32)
    img2 = cv2.warpAffine(img, M, (480, 360))

    p0 = tmp_path / "a.jpg"
    p1 = tmp_path / "b.jpg"
    cv2.imwrite(str(p0), img)
    cv2.imwrite(str(p1), img2)
    return str(p0), str(p1)


def _lightglue_available() -> bool:
    try:
        import lightglue  # noqa: F401
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not _lightglue_available(),
    reason="lightglue not installed (uv pip install -e models/LightGlue)",
)
def test_lightglue_matcher_returns_finite_matches(tmp_path: Path) -> None:
    warnings.filterwarnings("ignore")
    from localization_sensor_fusion._internal.matching.lightglue import (
        LightGlueMatcher,
    )

    p0, p1 = _synth_pair(tmp_path)
    matcher: FeatureMatcher = LightGlueMatcher(max_num_keypoints=512)
    assert matcher.backend_name == "lightglue"

    result: MatchResult = matcher.match(p0, p1)

    assert isinstance(result, MatchResult)
    assert result.points0.shape == result.points1.shape
    assert result.points0.ndim == 2 and result.points0.shape[1] == 2
    assert result.n_matches >= 20, (
        f"expected a usable number of matches on synthetic overlap, "
        f"got {result.n_matches}"
    )
    assert np.isfinite(result.points0).all()
    assert np.isfinite(result.points1).all()
    # Coordinates live in the LightGlue-resized image space, which by
    # default has a 1024-pixel long side, so we only assert that they
    # are non-negative and have a sane extent.
    assert (result.points0 >= 0).all() and (result.points1 >= 0).all()
    assert result.points0.max() > 100 and result.points1.max() > 100
    if result.scores is not None:
        assert np.isfinite(result.scores).all()
        assert (result.scores >= 0).all() and (result.scores <= 1.0 + 1e-6).all()


@pytest.mark.skipif(
    not _lightglue_available(),
    reason="lightglue not installed (uv pip install -e models/LightGlue)",
)
def test_build_matcher_lightglue_backend(tmp_path: Path) -> None:
    warnings.filterwarnings("ignore")
    matcher = build_matcher({"backend": "lightglue", "max_num_keypoints": 256})
    assert matcher.backend_name == "lightglue"

    p0, p1 = _synth_pair(tmp_path, seed=1)
    result = matcher.match(p0, p1)
    assert result.points0.shape == result.points1.shape
    assert np.isfinite(result.points0).all()


def test_build_matcher_classical_default() -> None:
    matcher = build_matcher(None)
    assert matcher.backend_name == "classical"

    matcher2 = build_matcher({"backend": "classical"})
    assert matcher2.backend_name == "classical"


def test_build_matcher_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="Unknown matcher backend"):
        build_matcher({"backend": "definitely-not-a-real-backend"})


def test_match_result_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        MatchResult(
            points0=np.zeros((3, 2)),
            points1=np.zeros((4, 2)),
        )
