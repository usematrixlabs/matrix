"""S1 Metadata Extractor.

Extracts structured metadata from video streams, timing models, optional camera/flight/sensor
metadata, and camera calibration parameters. Conforms to Phase 3 & Phase 9 deliverables.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from .camera_calibrator import CameraCalibrationLoader
from .exceptions import VideoMetadataError
from .logger import get_logger
from .types import (
    CameraCalibration,
    CameraMetadata,
    FlightMetadata,
    FrameTimingInfo,
    SensorMetadata,
    VideoMetadata,
    VideoMetadataRecord,
)
from .video_validator import VideoValidator


class MetadataExtractor:
    """Extracts stream properties, timing parameters, and sidecar metadata."""

    def __init__(self, log_level: str = "INFO"):
        """Initialize the metadata extractor.

        Parameters:
            log_level (str): Logging level.
        """
        self.logger = get_logger(self.__class__.__name__, log_level=log_level)
        self.validator = VideoValidator(log_level=log_level)
        self.calibration_loader = CameraCalibrationLoader()

    def extract(
        self,
        video_path: str,
        sidecar_path: Optional[str] = None,
        sidecar_data: Optional[Dict[str, Any]] = None,
        calibration_path: Optional[str] = None,
        start_time_offset: float = 0.0,
    ) -> VideoMetadataRecord:
        """Extract complete structured metadata for a UAV video source.

        Parameters:
            video_path (str): Path to the video file.
            sidecar_path (Optional[str]): Path to auxiliary metadata JSON/dictionary.
            sidecar_data (Optional[Dict[str, Any]]): Pre-loaded dictionary of sidecar metadata.
            calibration_path (Optional[str]): Path to camera calibration JSON/YAML.
            start_time_offset (float): Initial timestamp offset in seconds.

        Returns:
            VideoMetadataRecord: Structured video and sensor metadata representation.

        Raises:
            VideoValidationError: If the source video fails validation checks.
        """
        self.logger.info("Extracting video metadata for '%s'...", video_path)

        # 1. Validate stream and extract fundamental stream properties
        video_meta = self.validator.validate(video_path)

        # 2. Build monotonic FrameTimingInfo
        if video_meta.fps <= 0:
            raise VideoMetadataError(f"Cannot calculate timing info with invalid FPS: {video_meta.fps}")

        frame_interval = round(1.0 / video_meta.fps, 6)
        timing_info = FrameTimingInfo(
            fps=video_meta.fps,
            frame_interval_seconds=frame_interval,
            total_frames=video_meta.frame_count,
            start_timestamp=start_time_offset,
            end_timestamp=round(start_time_offset + video_meta.duration_seconds, 6),
            duration_seconds=video_meta.duration_seconds,
        )

        # 3. Load optional sidecar metadata
        raw_sidecar: Dict[str, Any] = {}
        if sidecar_data:
            raw_sidecar = sidecar_data
        elif sidecar_path and os.path.exists(sidecar_path):
            try:
                with open(sidecar_path, "r", encoding="utf-8") as f:
                    raw_sidecar = json.load(f)
                self.logger.info("Loaded auxiliary sidecar metadata from '%s'", sidecar_path)
            except Exception as e:
                self.logger.warning("Failed to load sidecar metadata '%s': %s", sidecar_path, e)
        elif sidecar_path:
            self.logger.warning("Sidecar metadata path '%s' not found.", sidecar_path)

        # 4. Extract Camera Calibration (Phase 9)
        calib_source = (
            calibration_path
            or raw_sidecar.get("calibration")
            or raw_sidecar.get("camera_parameters")
            or raw_sidecar.get("camera_calibration")
        )
        calibration_meta = self.calibration_loader.load_calibration(
            calibration_source=calib_source,
            image_width=video_meta.width,
            image_height=video_meta.height,
        )

        # 5. Extract Camera Metadata (explicitly None if absent)
        camera_dict = raw_sidecar.get("camera", {})
        camera_meta = CameraMetadata(
            camera_id=camera_dict.get("camera_id", "primary"),
            camera_make=camera_dict.get("camera_make") or camera_dict.get("make"),
            camera_model=camera_dict.get("camera_model") or camera_dict.get("model"),
            focal_length_mm=camera_dict.get("focal_length_mm"),
            sensor_width_mm=camera_dict.get("sensor_width_mm"),
            sensor_height_mm=camera_dict.get("sensor_height_mm"),
            field_of_view_deg=camera_dict.get("field_of_view_deg"),
            lens_parameters=camera_dict.get("lens_parameters"),
            exposure_mode=camera_dict.get("exposure_mode"),
            calibration=calibration_meta,
        )

        # 6. Extract Flight Metadata (explicitly None if absent)
        flight_dict = raw_sidecar.get("flight", {})
        flight_meta = FlightMetadata(
            flight_id=flight_dict.get("flight_id"),
            aircraft_model=flight_dict.get("aircraft_model") or flight_dict.get("drone_model"),
            takeoff_timestamp=flight_dict.get("takeoff_timestamp"),
            pilot_operator=flight_dict.get("pilot_operator") or flight_dict.get("operator"),
            mission_type=flight_dict.get("mission_type"),
        )

        # 7. Extract Sensor Metadata (explicitly None/False if absent)
        sensor_dict = raw_sidecar.get("sensor", {}) or raw_sidecar.get("sensors", {})
        has_gps = sensor_dict.get("has_gps", bool(raw_sidecar.get("gps_coordinates") or raw_sidecar.get("gps")))
        has_imu = sensor_dict.get("has_imu", bool(raw_sidecar.get("imu_data") or raw_sidecar.get("imu")))
        has_rtk = sensor_dict.get("has_rtk", bool(raw_sidecar.get("rtk_ppk") or raw_sidecar.get("rtk")))

        sensor_meta = SensorMetadata(
            has_gps=has_gps,
            has_imu=has_imu,
            has_rtk=has_rtk,
            gps_sampling_rate_hz=sensor_dict.get("gps_sampling_rate_hz"),
            imu_sampling_rate_hz=sensor_dict.get("imu_sampling_rate_hz"),
            coordinate_system=sensor_dict.get("coordinate_system", "WGS84"),
            altitude_reference=sensor_dict.get("altitude_reference"),
        )

        record = VideoMetadataRecord(
            video=video_meta,
            timing=timing_info,
            camera=camera_meta,
            flight=flight_meta,
            sensor=sensor_meta,
            calibration=calibration_meta,
            source_file=str(Path(video_path).resolve()),
        )

        self.logger.info(
            "Video metadata extraction complete: %dx%d, %.2f FPS, %d frames (dt=%.4fs, is_calibrated=%s)",
            video_meta.width,
            video_meta.height,
            video_meta.fps,
            video_meta.frame_count,
            timing_info.frame_interval_seconds,
            calibration_meta.is_calibrated,
        )
        return record
