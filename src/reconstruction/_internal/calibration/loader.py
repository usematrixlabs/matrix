"""OpenCV-format camera calibration loader.

Parses calibration files written by OpenCV's
``cv2.FileStorage`` (``%YAML:1.0`` header, ``!!opencv-matrix``
custom tags). The accepted shape is::

    %YAML:1.0
    ---
    camera_name: DJI_Air_2S
    image_width: 1920
    image_height: 1080
    distortion_model: plumb_bob
    camera_matrix: !!opencv-matrix
       rows: 3
       cols: 3
       dt: d
       data: [ fx, 0, cx,
               0, fy, cy,
               0,  0,  1 ]
    distortion_coefficients: !!opencv-matrix
       rows: 1
       cols: 5
       dt: d
       data: [ k1, k2, p1, p2, k3 ]

Validation is structural — every required field is checked and the
:class:`CameraCalibration` invariants are enforced in
:func:`CameraCalibration.validate`. Errors are surfaced as
:class:`CameraCalibrationError` with descriptive messages so a malformed
file does not silently degrade the pipeline.

The loader does not perform any resolution scaling — that is the
caller's responsibility (see :func:`CameraCalibration.scale_to_resolution`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from ..models.calibration import CameraCalibration, CameraCalibrationError


_OPENCV_MATRIX_TAG = "opencv-matrix"


class OpenCVCameraCalibrationLoader:
    """Loader for OpenCV ``FileStorage`` YAML calibration files."""

    @staticmethod
    def load_from_file(path: Union[str, Path]) -> CameraCalibration:
        """Load and validate a calibration from an OpenCV YAML file.

        Raises :class:`CameraCalibrationError` on any malformed input.
        """
        p = Path(path)
        if not p.is_file():
            raise CameraCalibrationError(
                f"Calibration file does not exist: {p}"
            )
        try:
            with open(p, "r", encoding="utf-8") as fh:
                raw_text = fh.read()
        except OSError as exc:
            raise CameraCalibrationError(
                f"Could not read calibration file {p}: {exc}"
            ) from exc

        try:
            data = _load_opencv_yaml(raw_text)
        except yaml.YAMLError as exc:
            raise CameraCalibrationError(
                f"Could not parse YAML in calibration file {p}: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise CameraCalibrationError(
                f"Calibration file {p} must deserialize to a mapping, "
                f"got {type(data).__name__}."
            )

        try:
            calib = OpenCVCameraCalibrationLoader._from_dict(data, source=str(p))
        except CameraCalibrationError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise CameraCalibrationError(
                f"Unexpected error parsing calibration {p}: {exc}"
            ) from exc
        return calib

    @staticmethod
    def load_from_dict(data: Dict[str, Any]) -> CameraCalibration:
        """Load and validate a calibration from an in-memory dict.

        This is useful for testing and for callers that already have the
        YAML in memory (for example S1's metadata extraction pipeline).
        """
        return OpenCVCameraCalibrationLoader._from_dict(data, source=None)

    @staticmethod
    def _from_dict(data: Dict[str, Any], source: Optional[str]) -> CameraCalibration:
        """Internal construction + validation."""
        if not isinstance(data, dict):
            raise CameraCalibrationError(
                f"Calibration data must be a mapping, got {type(data).__name__}."
            )

        camera_name = data.get("camera_name")
        if camera_name is None:
            raise CameraCalibrationError(
                "Calibration is missing required field 'camera_name'."
            )
        if not isinstance(camera_name, str):
            raise CameraCalibrationError(
                f"camera_name must be a string, got {type(camera_name).__name__}."
            )

        image_width = _require_int(data, "image_width")
        image_height = _require_int(data, "image_height")

        distortion_model = data.get("distortion_model")
        if distortion_model is None:
            raise CameraCalibrationError(
                "Calibration is missing required field 'distortion_model'."
            )
        if not isinstance(distortion_model, str):
            raise CameraCalibrationError(
                f"distortion_model must be a string, got {type(distortion_model).__name__}."
            )

        if "camera_matrix" not in data:
            raise CameraCalibrationError(
                "Calibration is missing required field 'camera_matrix'."
            )
        K = _parse_opencv_matrix(data["camera_matrix"], expected_shape=(3, 3), field="camera_matrix")

        if "distortion_coefficients" not in data:
            raise CameraCalibrationError(
                "Calibration is missing required field 'distortion_coefficients'."
            )
        D_raw = _parse_opencv_matrix(
            data["distortion_coefficients"],
            expected_shape=None,  # accept either (1, N) or (N, 1) or (N,)
            field="distortion_coefficients",
        )
        D = np.asarray(D_raw, dtype=np.float64).reshape(-1)

        # Preserve non-required fields under ``extra`` so callers can
        # audit the original file content without us silently dropping
        # data.
        known_keys = {
            "camera_name",
            "image_width",
            "image_height",
            "distortion_model",
            "camera_matrix",
            "distortion_coefficients",
        }
        extra = {k: v for k, v in data.items() if k not in known_keys}

        calib = CameraCalibration(
            camera_name=str(camera_name),
            image_width=int(image_width),
            image_height=int(image_height),
            distortion_model=str(distortion_model),
            camera_matrix=K,
            distortion_coefficients=D,
            source=source,
            extra=extra,
        )
        calib.validate()
        return calib


def _require_int(data: Dict[str, Any], key: str) -> int:
    if key not in data:
        raise CameraCalibrationError(
            f"Calibration is missing required field {key!r}."
        )
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int):
        try:
            value = int(value)
        except (TypeError, ValueError) as exc:
            raise CameraCalibrationError(
                f"Calibration field {key!r} must be an integer, got {value!r}."
            ) from exc
    if value <= 0:
        raise CameraCalibrationError(
            f"Calibration field {key!r} must be a positive integer, got {value}."
        )
    return int(value)


def _parse_opencv_matrix(value: Any, expected_shape: Optional[tuple], field: str) -> np.ndarray:
    """Parse an OpenCV ``!!opencv-matrix`` node.

    After ``yaml.safe_load`` the matrix tag has already been stripped —
    the node is either a flat list ``[a, b, c, ...]`` (when written by
    OpenCV with single-row data) or a ``{rows, cols, dt, data}`` dict
    (the canonical form). Both shapes are supported here.
    """
    arr: Optional[np.ndarray] = None

    if isinstance(value, dict):
        if "data" not in value:
            raise CameraCalibrationError(
                f"{field}: opencv-matrix dict is missing 'data' key: {value!r}"
            )
        try:
            data_list = [float(v) for v in value["data"]]
        except (TypeError, ValueError) as exc:
            raise CameraCalibrationError(
                f"{field}: opencv-matrix 'data' must be numeric, got {value['data']!r}"
            ) from exc
        rows = value.get("rows")
        cols = value.get("cols")
        if rows is not None and cols is not None:
            try:
                arr = np.asarray(data_list, dtype=np.float64).reshape(int(rows), int(cols))
            except ValueError as exc:
                raise CameraCalibrationError(
                    f"{field}: cannot reshape data ({len(data_list)} elements) "
                    f"to ({rows}, {cols}): {exc}"
                ) from exc
        else:
            arr = np.asarray(data_list, dtype=np.float64).reshape(1, -1)
    elif isinstance(value, list):
        flat = []
        for elem in value:
            if isinstance(elem, list):
                flat.extend(_flatten_numeric(elem))
            else:
                flat.extend(_flatten_numeric([elem]))
        if not flat:
            raise CameraCalibrationError(
                f"{field}: opencv-matrix list is empty."
            )
        arr = np.asarray(flat, dtype=np.float64).reshape(1, -1)
    else:
        raise CameraCalibrationError(
            f"{field}: expected opencv-matrix dict or list, got {type(value).__name__}."
        )

    if arr is None:
        raise CameraCalibrationError(f"{field}: could not parse opencv-matrix.")

    if expected_shape is not None and arr.shape != expected_shape:
        # For the camera matrix we strictly require 3x3; for distortion
        # we accept any 1D row/column.
        raise CameraCalibrationError(
            f"{field}: expected shape {expected_shape}, got {arr.shape}."
        )

    return arr


def _flatten_numeric(seq: List[Any]) -> List[float]:
    out: List[float] = []
    for item in seq:
        if isinstance(item, list):
            out.extend(_flatten_numeric(item))
        else:
            try:
                out.append(float(item))
            except (TypeError, ValueError) as exc:
                raise CameraCalibrationError(
                    f"Opencv-matrix element is not numeric: {item!r}"
                ) from exc
    return out


__all__ = ["OpenCVCameraCalibrationLoader"]


# Local numpy import kept at module bottom to avoid polluting the
# module-level namespace if this loader is imported very early.
import numpy as np  # noqa: E402


class _OpenCVYamlLoader(yaml.SafeLoader):
    """SafeLoader variant that ignores unknown YAML tags (e.g. ``!opencv-matrix``).

    OpenCV writes its calibration files with a ``%YAML:1.0`` directive
    and various custom tag shorthands for matrices. Both need light
    customization to parse under modern PyYAML: the directive must be
    stripped (PyYAML 6.x rejects ``%YAML:1.0`` as not matching its
    expected ``%YAML`` token) and the custom tag must be mapped to a
    plain mapping constructor so we can interpret the matrix structure
    ourselves. We register all observed shorthands:

    * ``!opencv-matrix`` / ``!!opencv-matrix`` — primary local tag form
    * ``!x!opencv-matrix`` — OpenCV's primary-document shorthand that
      reuses the tag prefix registered on the first ``!!opencv-matrix``
      occurrence
    """


def _construct_opencv_matrix(loader: "_OpenCVYamlLoader", node: yaml.MappingNode) -> Dict[str, Any]:
    return loader.construct_mapping(node, deep=True)


for _tag in (
    "!opencv-matrix",
    "!!opencv-matrix",
    "!x!opencv-matrix",
    "tag:yaml.org,2002:opencv-matrix",
):
    _OpenCVYamlLoader.add_constructor(_tag, _construct_opencv_matrix)


def _load_opencv_yaml(text: str) -> Any:
    """Parse an OpenCV FileStorage YAML document.

    OpenCV's ``cv2.FileStorage`` writes a YAML 1.0 file with a custom
    tag shorthand for matrices. To make the file parseable with modern
    PyYAML (6.x+) we:

    1. Strip the ``%YAML:1.0`` directive — PyYAML 6.x rejects the
       non-alphabetic version token.
    2. Inject ``%TAG !x! tag:yaml.org,2002:`` so the ``!x!opencv-matrix``
       shorthand that OpenCV writes on the second matrix becomes a
       defined tag handle.
    3. Delegate to a SafeLoader subclass that maps all known OpenCV
       matrix tag shorthands to a plain mapping constructor.
    """
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("%YAML"):
            continue
        if stripped.startswith("%"):
            continue
        cleaned_lines.append(line)

    if "!x!" in "\n".join(cleaned_lines):
        cleaned_lines.insert(0, "%TAG !x! tag:yaml.org,2002:")

    cleaned = "\n".join(cleaned_lines)
    return yaml.load(cleaned, Loader=_OpenCVYamlLoader)
