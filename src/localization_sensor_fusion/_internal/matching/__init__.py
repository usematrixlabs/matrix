"""S2 visual-correspondence (feature-matching) backend abstraction.

This subpackage isolates the choice of feature detector/matcher from the
rest of S2. Pose estimation, trajectory smoothing, sensor fusion, and the
S2 contract are all unaware of which backend is in use; they only see a
:class:`FeatureMatcher` that returns a Matrix-native
:class:`MatchResult` of pixel correspondences.

Backends currently provided:

* :class:`ClassicalMatcher` -- ORB + BFMatcher, the historical default.
* :class:`LightGlueMatcher` -- SuperPoint + LightGlue (vendored under
  ``models/LightGlue``). Optional import: this module is only required
  when the user actually selects ``backend: lightglue``.

Selection is configuration-driven via :func:`build_matcher`.
"""

from .base import FeatureMatcher, MatchResult, build_matcher

__all__ = ["FeatureMatcher", "MatchResult", "build_matcher"]
