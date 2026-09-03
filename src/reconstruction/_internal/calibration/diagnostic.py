"""Camera calibration diagnostic.

Reads an OpenCV ``%YAML:1.0`` calibration file, validates it, and prints
a human-readable summary plus an optional before/after feature-point
visualization. Useful at hackathon time to confirm that a calibration
file is well-formed and that ``cv2.undistortPoints`` is moving feature
points in the expected direction.

CLI:

    python -m src.reconstruction._internal.calibration.diagnostic \\
        path/to/camera_calibration.yaml [--visualize path/to/image.png]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from ..models.calibration import CameraCalibration, CameraCalibrationError
from .loader import OpenCVCameraCalibrationLoader


def _print_calibration(calib: CameraCalibration, image_path: Optional[Path]) -> None:
    print("=" * 64)
    print(f"Camera:        {calib.camera_name}")
    print(f"Resolution:    {calib.image_width} × {calib.image_height}")
    print(f"Distortion:    {calib.distortion_model}")
    print(f"Source:        {calib.source or '<unknown>'}")
    print("-" * 64)
    print("Intrinsics:")
    print(f"  fx = {calib.fx:.6f}")
    print(f"  fy = {calib.fy:.6f}")
    print(f"  cx = {calib.cx:.6f}")
    print(f"  cy = {calib.cy:.6f}")
    print("Camera matrix K:")
    for row in calib.camera_matrix.tolist():
        print(f"  {row}")
    print("Distortion coefficients:")
    coeffs = calib.distortion_coefficients.tolist()
    print(f"  [{', '.join(f'{c:.10f}' for c in coeffs)}]")
    if len(coeffs) == 5:
        print("  (k1, k2, p1, p2, k3) — OpenCV plumb_bob / radtan")
    print("-" * 64)
    print("Undistortion:")
    print("  enabled = True")
    print("  cv2.undistortPoints(pts, K, D, P=K) returns undistorted")
    print("  pixel coordinates in the same K pixel space, suitable for")
    print("  P = K [R | t] projection in the S3 triangulation pipeline.")
    if image_path is not None:
        print("-" * 64)
        print(f"Visualization image: {image_path}")
    print("=" * 64)


def _maybe_visualize(calib: CameraCalibration, image_path: Optional[Path], out_path: Path) -> None:
    try:
        import cv2
    except ImportError:
        print("(skipping visualization — cv2 not available)")
        return
    if image_path is None or not Path(image_path).is_file():
        print("(skipping visualization — no image provided)")
        return

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"(skipping visualization — could not read {image_path})")
        return

    h, w = img.shape[:2]
    if (w, h) != (calib.image_width, calib.image_height):
        print(
            f"(warning: image is {w}x{h}, calibration is "
            f"{calib.image_width}x{calib.image_height}; using image-native scale)"
        )
        calib = calib.scale_to_resolution(w, h)

    grid = np.mgrid[
        slice(0.1 * h, 0.9 * h, 14j),
        slice(0.1 * w, 0.9 * w, 24j),
    ].reshape(2, -1).T
    pts = np.stack([grid[:, 1], grid[:, 0]], axis=1).astype(np.float64)

    K = calib.camera_matrix.astype(np.float64)
    D = calib.distortion_coefficients.reshape(-1, 1).astype(np.float64)
    und = cv2.undistortPoints(
        pts.reshape(-1, 1, 2), K, D, P=K,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-6),
    ).reshape(-1, 2)

    overlay = img.copy()
    for (u_d, v_d) in pts:
        cv2.circle(overlay, (int(round(u_d)), int(round(v_d))), 4, (0, 0, 255), -1)  # red = raw
    for (u_u, v_u) in und:
        cv2.circle(overlay, (int(round(u_u)), int(round(v_u))), 3, (0, 255, 0), 1)  # green = undistorted
    cv2.imwrite(str(out_path), overlay)
    print(f"Wrote before/after visualization to: {out_path}")


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Camera calibration diagnostic for OpenCV YAML files."
    )
    parser.add_argument("calibration", type=Path, help="OpenCV YAML calibration file")
    parser.add_argument(
        "--visualize", type=Path, default=None,
        help="Optional frame image to overlay raw vs. undistorted feature points.",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("calibration_visualization.png"),
        help="Output path for the visualization image.",
    )
    args = parser.parse_args(argv)

    try:
        calib: CameraCalibration = OpenCVCameraCalibrationLoader.load_from_file(args.calibration)
    except CameraCalibrationError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    _print_calibration(calib, args.visualize)
    _maybe_visualize(calib, args.visualize, args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
