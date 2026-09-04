"""S3 Camera Calibration Model.

Generic camera calibration representation for S3 (3D Reconstruction).

This model is the canonical "calibration input" object consumed by the
S3 reconstruction pipeline. It is intentionally camera-agnostic — the
specific DJI Air 2S calibration is just one possible input file, and
no production code path should ever reference the supplied values
explicitly.

The model exposes:

- ``camera_name``         : human-readable identifier (e.g. "DJI_Air_2S")
- ``image_width``         : calibration image width (pixels)
- ``image_height``        : calibration image height (pixels)
- ``distortion_model``    : e.g. "plumb_bob", "radtan", "fisheye"
- ``camera_matrix``       : 3x3 ``numpy.ndarray`` pinhole intrinsic matrix K
- ``distortion_coefficients`` : ``numpy.ndarray`` of length 5/8/etc. per
                                the model

It does **not** subsume :class:`CameraIntrinsics`; that lightweight
internal model is what :class:`CameraPose.projection_matrix` consumes
because projection only needs ``K``. The conversion
:func:`CameraCalibration.to_camera_intrinsics` projects the rich
calibration record down to the minimal ``CameraIntrinsics`` while
preserving the original ``CameraCalibration`` for downstream metadata.

Coordinate convention
---------------------
This module follows the OpenCV / S3 convention exactly:

- Distortion model ``plumb_bob`` ⇒ ``[k1, k2, p1, p2, k3]`` in that
  order, applied with ``cv2.projectPoints`` / ``cv2.undistortPoints``.
- ``camera_matrix`` is the standard pinhole matrix::

      | fx  0  cx |
      |  0 fy  cy |
      |  0  0   1 |

- Pixel coordinates are 0-indexed, with origin at the top-left of the
  image. ``cx``/``cy`` are measured from that origin.

The model does **not** silently reinterpret, normalize, clamp, or
"correct" any coefficient values — the supplied calibration is
consumed exactly as given. Concerns about calibration quality must be
surfaced as a validation warning, never silently mutated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .schema import CameraIntrinsics


_SUPPORTED_DISTORTION_MODELS = frozenset({"plumb_bob", "radtan"})

_PLUMB_BOB_COEFFICIENT_COUNT = 5
_RADTAN_COEFFICIENT_COUNT = 5
_FISHEYE_COEFFICIENT_COUNT = 4


class CameraCalibrationError(ValueError):
    """Raised when a camera calibration file or object is malformed."""


@dataclass
class CameraCalibration:
    """Generic camera calibration record consumed by S3.

    Attributes
    ----------
    camera_name : str
        Human-readable camera identifier (e.g. ``"DJI_Air_2S"``).
    image_width : int
        Calibration image width in pixels (calibration was performed
        at this resolution; scaling rules apply for different video
        resolutions).
    image_height : int
        Calibration image height in pixels.
    distortion_model : str
        One of the supported OpenCV distortion models. Currently
        ``"plumb_bob"`` and the alias ``"radtan"`` are accepted. Both
        use the OpenCV 5-coefficient ordering ``[k1, k2, p1, p2, k3]``.
    camera_matrix : numpy.ndarray
        ``(3, 3)`` float64 pinhole intrinsic matrix.
    distortion_coefficients : numpy.ndarray
        ``(N,)`` float64 distortion coefficient vector. Length is
        model-dependent (5 for ``plumb_bob``/``radtan``).
    source : Optional[str]
        Origin identifier (file path, sidecar name, etc.). For
        reproducibility only.
    extra : Dict[str, Any]
        Free-form additional fields preserved verbatim for debugging
        and downstream audit (e.g. rectification matrices, additional
        metadata).
    """

    camera_name: str
    image_width: int
    image_height: int
    distortion_model: str
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    source: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def fx(self) -> float:
        return float(self.camera_matrix[0, 0])

    @property
    def fy(self) -> float:
        return float(self.camera_matrix[1, 1])

    @property
    def cx(self) -> float:
        return float(self.camera_matrix[0, 2])

    @property
    def cy(self) -> float:
        return float(self.camera_matrix[1, 2])

    def validate(self) -> None:
        """Run structural validation on the calibration record.

        Raises :class:`CameraCalibrationError` if any field violates the
        documented invariants. The calibration file is consumed exactly
        as supplied — no normalization or "correction" of coefficient
        magnitudes is performed here.
        """
        if not isinstance(self.camera_name, str) or not self.camera_name:
            raise CameraCalibrationError(
                f"camera_name must be a non-empty string, got {self.camera_name!r}"
            )
        if not isinstance(self.image_width, int) or self.image_width <= 0:
            raise CameraCalibrationError(
                f"image_width must be a positive integer, got {self.image_width!r}"
            )
        if not isinstance(self.image_height, int) or self.image_height <= 0:
            raise CameraCalibrationError(
                f"image_height must be a positive integer, got {self.image_height!r}"
            )
        if self.distortion_model not in _SUPPORTED_DISTORTION_MODELS:
            raise CameraCalibrationError(
                f"Unsupported distortion_model {self.distortion_model!r}. "
                f"Supported models: {sorted(_SUPPORTED_DISTORTION_MODELS)}."
            )

        K = np.asarray(self.camera_matrix, dtype=np.float64)
        if K.shape != (3, 3):
            raise CameraCalibrationError(
                f"camera_matrix must be 3x3, got shape {K.shape}."
            )
        if not np.all(np.isfinite(K)):
            raise CameraCalibrationError("camera_matrix contains non-finite values.")
        if not np.isclose(K[2, 2], 1.0, atol=1e-6):
            raise CameraCalibrationError(
                f"camera_matrix[2,2] must be ≈ 1.0 (pinhole convention); got {K[2, 2]!r}."
            )
        if K[0, 1] != 0.0 or K[1, 0] != 0.0 or K[2, 0] != 0.0 or K[2, 1] != 0.0:
            raise CameraCalibrationError(
                f"camera_matrix has non-zero off-diagonal lower/upper entries "
                f"(must be standard pinhole form): {K.tolist()}"
            )
        if K[0, 0] <= 0 or K[1, 1] <= 0:
            raise CameraCalibrationError(
                f"camera_matrix focal lengths must be positive; "
                f"fx={K[0, 0]!r}, fy={K[1, 1]!r}."
            )
        if not (0.0 <= K[0, 2] <= self.image_width):
            raise CameraCalibrationError(
                f"camera_matrix principal point cx={K[0, 2]!r} must lie in "
                f"[0, image_width={self.image_width}]."
            )
        if not (0.0 <= K[1, 2] <= self.image_height):
            raise CameraCalibrationError(
                f"camera_matrix principal point cy={K[1, 2]!r} must lie in "
                f"[0, image_height={self.image_height}]."
            )

        dist = np.asarray(self.distortion_coefficients, dtype=np.float64)
        if dist.ndim != 1:
            raise CameraCalibrationError(
                f"distortion_coefficients must be a 1D vector, got shape {dist.shape}."
            )
        if not np.all(np.isfinite(dist)):
            raise CameraCalibrationError(
                "distortion_coefficients contains non-finite values."
            )
        expected_len = _expected_coefficient_count(self.distortion_model)
        if dist.size != expected_len:
            raise CameraCalibrationError(
                f"distortion_model {self.distortion_model!r} requires exactly "
                f"{expected_len} coefficients, got {dist.size}."
            )

    def expected_coefficient_count(self) -> int:
        """Return the expected number of distortion coefficients for the model."""
        return _expected_coefficient_count(self.distortion_model)

    def to_camera_intrinsics(self) -> CameraIntrinsics:
        """Project this calibration down to S3's minimal :class:`CameraIntrinsics`.

        The lightweight ``CameraIntrinsics`` is what :class:`CameraPose.projection_matrix`
        consumes. We keep the rich :class:`CameraCalibration` separate so that
        the full provenance (camera_name, image dims, distortion) is preserved
        in :class:`S3ReconstructionResult.metadata`.
        """
        return CameraIntrinsics(
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            width=int(self.image_width),
            height=int(self.image_height),
            distortion_coefficients=self.distortion_coefficients.tolist(),
            distortion_model=self.distortion_model,
        )

    def scale_to_resolution(self, video_width: int, video_height: int) -> "CameraCalibration":
        """Return a new calibration scaled to a different video resolution.

        Assumes an **isotropic uniform resize** (no cropping). When the
        video resolution does not match the calibration resolution we do
        *not* silently use the calibration; instead we explicitly scale
        the intrinsics by ``sx = video_w / calib_w`` and
        ``sy = video_h / calib_h`` (and require ``sx ≈ sy`` so the
        assumption of isotropic resampling holds). Distortion
        coefficients are pixel-space quantities and are invariant under
        uniform scaling.

        Returns ``self`` (a copy) when the resolutions already match.
        """
        if (
            int(video_width) == int(self.image_width)
            and int(video_height) == int(self.image_height)
        ):
            return CameraCalibration(
                camera_name=self.camera_name,
                image_width=int(self.image_width),
                image_height=int(self.image_height),
                distortion_model=self.distortion_model,
                camera_matrix=self.camera_matrix.astype(np.float64, copy=True),
                distortion_coefficients=self.distortion_coefficients.astype(
                    np.float64, copy=True
                ),
                source=self.source,
                extra=dict(self.extra),
            )

        sx = float(video_width) / float(self.image_width)
        sy = float(video_height) / float(self.image_height)
        if not np.isclose(sx, sy, atol=1e-3):
            raise CameraCalibrationError(
                f"Calibration resolution {self.image_width}x{self.image_height} "
                f"cannot be uniformly scaled to non-uniform "
                f"{video_width}x{video_height} (sx={sx:.4f}, sy={sy:.4f}). "
                f"Provide a calibration that matches the video or apply "
                f"non-uniform scaling only after explicit user confirmation."
            )

        scaled_K = np.array(
            [
                [self.fx * sx, 0.0, self.cx * sx],
                [0.0, self.fy * sy, self.cy * sy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return CameraCalibration(
            camera_name=self.camera_name,
            image_width=int(video_width),
            image_height=int(video_height),
            distortion_model=self.distortion_model,
            camera_matrix=scaled_K,
            distortion_coefficients=self.distortion_coefficients.astype(
                np.float64, copy=True
            ),
            source=self.source,
            extra={**self.extra, "scaled_from_resolution": [self.image_width, self.image_height]},
        )

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly serialization of this calibration record."""
        return {
            "camera_name": self.camera_name,
            "image_width": int(self.image_width),
            "image_height": int(self.image_height),
            "distortion_model": self.distortion_model,
            "camera_matrix": np.asarray(self.camera_matrix, dtype=np.float64).tolist(),
            "distortion_coefficients": np.asarray(
                self.distortion_coefficients, dtype=np.float64
            ).tolist(),
            "source": self.source,
            "extra": dict(self.extra),
        }


def _expected_coefficient_count(model: str) -> int:
    if model == "plumb_bob":
        return _PLUMB_BOB_COEFFICIENT_COUNT
    if model == "radtan":
        return _RADTAN_COEFFICIENT_COUNT
    if model == "fisheye":
        return _FISHEYE_COEFFICIENT_COUNT
    return -1


__all__ = [
    "CameraCalibration",
    "CameraCalibrationError",
    "_SUPPORTED_DISTORTION_MODELS",
]
