"""S1 Camera Calibration Loader.

Parses, validates, and preserves camera intrinsic calibration and lens distortion parameters
(fx, fy, cx, cy, distortion coefficients) when available. Always guarantees known image
dimensions and gracefully represents missing calibration as explicit null values (is_calibrated=False)
without halting S1 execution.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .config import S1Config
from .logger import get_logger
from .types import CameraCalibration


class CameraCalibrationLoader:
    """Loads and formats camera intrinsic calibration parameters."""

    def __init__(self, config: Optional[S1Config] = None):
        """Initialize the calibration loader.

        Parameters:
            config (Optional[S1Config]): Subsystem configuration.
        """
        self.config = config or S1Config()
        self.logger = get_logger(self.__class__.__name__, log_level=self.config.log_level)

    def load_calibration(
        self,
        calibration_source: Optional[Union[str, Dict[str, Any]]] = None,
        image_width: int = 1920,
        image_height: int = 1080,
    ) -> CameraCalibration:
        """Load and parse camera calibration from a file path or dictionary.

        Parameters:
            calibration_source (Optional[Union[str, Dict[str, Any]]]): File path (JSON/YAML) or dictionary.
            image_width (int): Video image width in pixels (always known).
            image_height (int): Video image height in pixels (always known).

        Returns:
            CameraCalibration: Structured calibration record with explicit availability status.
        """
        # Ensure dimensions are valid positive integers
        w = max(1, int(image_width))
        h = max(1, int(image_height))

        if calibration_source is None:
            self.logger.debug("No camera calibration provided. Initializing uncalibrated record.")
            return CameraCalibration(width=w, height=h, is_calibrated=False)

        raw_data: Optional[Dict[str, Any]] = None
        source_name: Optional[str] = None

        if isinstance(calibration_source, dict):
            raw_data = calibration_source
            source_name = "dictionary"
        elif isinstance(calibration_source, str):
            p = Path(calibration_source)
            source_name = str(p)
            if not p.exists():
                self.logger.warning("Camera calibration file not found: '%s'. Continuing uncalibrated.", source_name)
                return CameraCalibration(width=w, height=h, is_calibrated=False, calibration_source=source_name)

            try:
                if p.suffix.lower() in {".yaml", ".yml"}:
                    try:
                        import yaml
                        with open(p, "r", encoding="utf-8") as f:
                            raw_data = yaml.safe_load(f)
                    except ImportError:
                        with open(p, "r", encoding="utf-8") as f:
                            raw_data = json.load(f)
                else:
                    with open(p, "r", encoding="utf-8") as f:
                        raw_data = json.load(f)
            except Exception as e:
                self.logger.warning("Could not parse calibration file '%s': %s. Continuing uncalibrated.", source_name, e)
                return CameraCalibration(width=w, height=h, is_calibrated=False, calibration_source=source_name)

        if not raw_data:
            return CameraCalibration(width=w, height=h, is_calibrated=False, calibration_source=source_name)

        # Parse intrinsics (supporting standard names and nested camera_matrix)
        fx: Optional[float] = None
        fy: Optional[float] = None
        cx: Optional[float] = None
        cy: Optional[float] = None
        dist_coeffs: Optional[List[float]] = None
        dist_model: Optional[str] = raw_data.get("distortion_model") or raw_data.get("model")
        cam_matrix: Optional[List[List[float]]] = None

        # Check for nested calibration keys
        data_block = raw_data.get("camera_parameters") or raw_data.get("calibration") or raw_data

        # Check for 3x3 camera matrix
        if "camera_matrix" in data_block and isinstance(data_block["camera_matrix"], list):
            matrix = data_block["camera_matrix"]
            if len(matrix) == 3 and all(len(row) == 3 for row in matrix):
                cam_matrix = [[float(v) for v in row] for row in matrix]
                fx = cam_matrix[0][0]
                fy = cam_matrix[1][1]
                cx = cam_matrix[0][2]
                cy = cam_matrix[1][2]
        elif "K" in data_block and isinstance(data_block["K"], list):
            matrix = data_block["K"]
            if len(matrix) == 3 and all(len(row) == 3 for row in matrix):
                cam_matrix = [[float(v) for v in row] for row in matrix]
                fx = cam_matrix[0][0]
                fy = cam_matrix[1][1]
                cx = cam_matrix[0][2]
                cy = cam_matrix[1][2]

        # Check individual focal/principal keys if not extracted from matrix
        if fx is None and "fx" in data_block:
            fx = float(data_block["fx"])
        if fy is None and "fy" in data_block:
            fy = float(data_block["fy"])
        if cx is None and "cx" in data_block:
            cx = float(data_block["cx"])
        if cy is None and "cy" in data_block:
            cy = float(data_block["cy"])

        # Check distortion coefficients
        for key in ["distortion_coefficients", "dist_coeffs", "distortion", "D"]:
            if key in data_block and isinstance(data_block[key], list):
                dist_coeffs = [float(v) for v in data_block[key]]
                break

        # Check dimension overrides if specified in file
        calib_w = int(data_block.get("width") or data_block.get("image_width") or w)
        calib_h = int(data_block.get("height") or data_block.get("image_height") or h)

        is_calibrated = (fx is not None and fy is not None and cx is not None and cy is not None)

        if is_calibrated and cam_matrix is None:
            cam_matrix = [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ]

        self.logger.info(
            "Loaded camera calibration (is_calibrated=%s, dimensions=%dx%d, source='%s')",
            is_calibrated,
            calib_w,
            calib_h,
            source_name,
        )

        return CameraCalibration(
            width=calib_w,
            height=calib_h,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_coefficients=dist_coeffs,
            distortion_model=dist_model,
            camera_matrix=cam_matrix,
            is_calibrated=is_calibrated,
            calibration_source=source_name,
        )

