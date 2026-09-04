"""LightGlue (SuperPoint + LightGlue) matcher backend.

This is an optional S2 backend. It is only imported when the user
selects ``backend: lightglue``; importing :mod:`base` does **not**
pull in Torch/LightGlue so CPU-only deployments that don't use the
backend don't pay the import cost.

All model specifics (torch device selection, preprocessing, ``rbd``
unbatching) are encapsulated here. Downstream S2 code only ever sees
a :class:`MatchResult`.

Coordinate-system note
----------------------
``lightglue.utils.load_image`` resizes images internally and the
returned keypoints live in the *resized* image's pixel coordinate
system. ``SuperPoint.extract`` does this consistently for both images,
so cross-image correspondences are coherent, but the absolute pixel
values will not match the original frame size. When feeding these
matches into the existing camera-model-based pipeline, downstream code
must either:

1. Rescale matches back to original image coordinates, or
2. Use the rescaled intrinsics for the resized image space.

For the SIH one-day sprint we deliberately do **not** perform custom
resizing here; if a later integration step needs original-resolution
keypoints, that mapping belongs in this module (not in pose estimation)
so the rest of S2 stays backend-agnostic.
"""
from __future__ import annotations

import os
from typing import Optional, Union

import numpy as np

from .base import MatchResult

PathOrImage = Union[str, np.ndarray]


class LightGlueMatcher:
    """SuperPoint + LightGlue matcher.

    Parameters
    ----------
    max_num_keypoints : int
        Maximum keypoints detected per image by SuperPoint. The official
        project documents 2048 as the standard example; we use that as
        the default. ``None`` disables the cap (slower but denser).
    device : str or None
        ``"cuda"``, ``"cpu"`` or ``None`` for auto-selection. Falls back
        to ``"cpu"`` if CUDA is unavailable.
    weights_dir : str or None
        Optional override for the torch hub cache directory. When
        ``None`` the default ``~/.cache/torch/hub`` is used, which keeps
        pretrained weights out of the repository.
    """

    backend_name = "lightglue"

    def __init__(
        self,
        max_num_keypoints: int = 2048,
        device: Optional[str] = None,
        weights_dir: Optional[str] = None,
    ) -> None:
        # Imported lazily so classical-only deployments don't pay the
        # Torch import cost. This is also where the "is lightglue even
        # available in this environment?" check lives.
        try:
            import torch  # noqa: F401
            from lightglue import LightGlue, SuperPoint  # noqa: F401
            from lightglue.utils import load_image, rbd  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "LightGlue backend requested but the 'lightglue' package "
                "is not installed. Install it with:\n"
                "    uv pip install -e models/LightGlue\n"
                "(from the repository root)."
            ) from exc

        self._torch = __import__("torch")
        self._lg_LightGlue = LightGlue
        self._lg_SuperPoint = SuperPoint
        self._lg_load_image = load_image
        self._lg_rbd = rbd

        if device is None:
            device = "cuda" if self._torch.cuda.is_available() else "cpu"
        self._device = self._torch.device(device)

        if weights_dir is not None:
            os.environ.setdefault("TORCH_HOME", weights_dir)

        self._extractor = (
            self._lg_SuperPoint(max_num_keypoints=int(max_num_keypoints))
            .eval()
            .to(self._device)
        )
        self._matcher = (
            self._lg_LightGlue(features="superpoint")
            .eval()
            .to(self._device)
        )

    @property
    def device(self) -> str:
        return str(self._device)

    def match(self, image0: PathOrImage, image1: PathOrImage) -> MatchResult:
        if not isinstance(image0, str):
            raise TypeError(
                "LightGlueMatcher currently only accepts filesystem paths; "
                "in-memory image arrays should be written by the caller."
            )
        if not isinstance(image1, str):
            raise TypeError(
                "LightGlueMatcher currently only accepts filesystem paths; "
                "in-memory image arrays should be written by the caller."
            )

        load_image = self._lg_load_image
        rbd = self._lg_rbd

        with self._torch.inference_mode():
            img0 = load_image(image0).to(self._device)
            img1 = load_image(image1).to(self._device)

            feats0 = self._extractor.extract(img0)
            feats1 = self._extractor.extract(img1)

            matches01 = self._matcher({"image0": feats0, "image1": feats1})

            feats0, feats1, matches01 = (rbd(x) for x in (feats0, feats1, matches01))

            idx = matches01["matches"]
            scores = matches01.get("scores", None)

        if idx is None or len(idx) == 0:
            return MatchResult(
                points0=np.empty((0, 2), dtype=np.float64),
                points1=np.empty((0, 2), dtype=np.float64),
                scores=np.empty((0,), dtype=np.float32) if scores is not None else None,
            )

        pts0 = feats0["keypoints"][idx[:, 0]].detach().cpu().numpy().astype(np.float64)
        pts1 = feats1["keypoints"][idx[:, 1]].detach().cpu().numpy().astype(np.float64)
        sc: Optional[np.ndarray] = None
        if scores is not None:
            sc = scores.detach().cpu().numpy().astype(np.float32)

        return MatchResult(points0=pts0, points1=pts1, scores=sc)


__all__ = ["LightGlueMatcher"]
