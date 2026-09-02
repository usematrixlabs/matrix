"""Unit tests for S1 Camera Metadata & Calibration (Phase 9).

Verifies that image dimensions (width, height) are always known, camera intrinsics
(fx, fy, cx, cy, distortion coefficients) are preserved when supplied, and missing
calibration is represented with explicit null values without halting S1 processing.
"""

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.visual_perception import S1Config, S1Output
from src.visual_perception._internal.types import CameraCalibration
from src.visual_perception._internal.camera_calibrator import CameraCalibrationLoader
from src.visual_perception._internal.metadata_extractor import MetadataExtractor
from src.visual_perception._internal.pipeline import S1Pipeline


def create_calib_test_video(file_path: str, width: int = 1280, height: int = 720, fps: float = 30.0) -> str:
    """Helper to synthesize a test video with known geometry."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
    for i in range(10):
        frame = np.full((height, width, 3), (i * 20) % 255, dtype=np.uint8)
        out.write(frame)
    out.release()
    return file_path


class TestS1CameraMetadata(unittest.TestCase):
    """Tests for CameraCalibrationLoader and calibration preservation."""

    def setUp(self):
        """Setup temporary test directory and video."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        self.video_path = str(self.temp_path / "uav_calib_test.mp4")
        create_calib_test_video(self.video_path, width=1280, height=720, fps=30.0)

    def tearDown(self):
        """Cleanup temporary files."""
        self.temp_dir.cleanup()

    def test_dimensions_always_known_without_calibration(self):
        """Verify width and height are always known from video stream even with no calibration."""
        extractor = MetadataExtractor()
        record = extractor.extract(video_path=self.video_path)

        self.assertEqual(record.video.width, 1280)
        self.assertEqual(record.video.height, 720)
        self.assertIsNotNone(record.calibration)
        self.assertEqual(record.calibration.width, 1280)
        self.assertEqual(record.calibration.height, 720)
        self.assertFalse(record.calibration.is_calibrated)
        self.assertIsNone(record.calibration.fx)
        self.assertIsNone(record.calibration.fy)
        self.assertIsNone(record.calibration.cx)
        self.assertIsNone(record.calibration.cy)
        self.assertIsNone(record.calibration.distortion_coefficients)

    def test_calibration_preserved_when_supplied_json(self):
        """Verify intrinsics and distortion are parsed and stored when valid JSON calibration is provided."""
        calib_file = str(self.temp_path / "camera_intrinsics.json")
        calib_data = {
            "width": 1280,
            "height": 720,
            "fx": 1150.5,
            "fy": 1152.8,
            "cx": 640.0,
            "cy": 360.0,
            "distortion_coefficients": [-0.15, 0.08, 0.001, -0.002, 0.0],
            "distortion_model": "radtan",
        }
        with open(calib_file, "w", encoding="utf-8") as f:
            json.dump(calib_data, f)

        extractor = MetadataExtractor()
        record = extractor.extract(video_path=self.video_path, calibration_path=calib_file)

        calib = record.calibration
        self.assertIsNotNone(calib)
        self.assertTrue(calib.is_calibrated)
        self.assertEqual(calib.fx, 1150.5)
        self.assertEqual(calib.fy, 1152.8)
        self.assertEqual(calib.cx, 640.0)
        self.assertEqual(calib.cy, 360.0)
        self.assertEqual(calib.distortion_coefficients, [-0.15, 0.08, 0.001, -0.002, 0.0])
        self.assertEqual(calib.distortion_model, "radtan")
        self.assertEqual(calib.camera_matrix, [[1150.5, 0.0, 640.0], [0.0, 1152.8, 360.0], [0.0, 0.0, 1.0]])

    def test_calibration_preserved_when_supplied_matrix_format(self):
        """Verify 3x3 camera matrix format is parsed correctly."""
        loader = CameraCalibrationLoader()
        matrix_dict = {
            "camera_matrix": [
                [1000.0, 0.0, 500.0],
                [0.0, 1000.0, 300.0],
                [0.0, 0.0, 1.0],
            ],
            "dist_coeffs": [-0.1, 0.01],
            "model": "pinhole",
        }
        calib = loader.load_calibration(calibration_source=matrix_dict, image_width=1000, image_height=600)

        self.assertTrue(calib.is_calibrated)
        self.assertEqual(calib.fx, 1000.0)
        self.assertEqual(calib.fy, 1000.0)
        self.assertEqual(calib.cx, 500.0)
        self.assertEqual(calib.cy, 300.0)
        self.assertEqual(calib.distortion_coefficients, [-0.1, 0.01])
        self.assertEqual(calib.distortion_model, "pinhole")

    def test_missing_calibration_does_not_stop_s1(self):
        """Verify missing calibration file does not halt S1 pipeline execution."""
        missing_path = str(self.temp_path / "non_existent_calibration.json")
        config = S1Config(
            video_path=self.video_path,
            calibration_path=missing_path,
            output_dir=str(self.temp_path / "pipe_out"),
            sampling_mode="fixed",
            sampling_interval=5,
        )
        pipeline = S1Pipeline(config=config)
        s1_out = pipeline.run()

        self.assertEqual(s1_out.status, "completed")
        self.assertGreater(len(s1_out.visual_observations.frames), 0)

        # Calibration must be explicit null
        calib_dict = s1_out.metadata.get("camera_calibration")
        self.assertIsNotNone(calib_dict)
        self.assertFalse(calib_dict["is_calibrated"])
        self.assertEqual(calib_dict["width"], 1280)
        self.assertEqual(calib_dict["height"], 720)
        self.assertIsNone(calib_dict["fx"])
        self.assertIsNone(calib_dict["fy"])

    def test_explicit_null_serialization(self):
        """Verify CameraCalibration.to_dict() contains explicit keys with null values when uncalibrated."""
        calib = CameraCalibration(width=1920, height=1080, is_calibrated=False)
        d = calib.to_dict()

        self.assertEqual(d["width"], 1920)
        self.assertEqual(d["height"], 1080)
        self.assertFalse(d["is_calibrated"])
        self.assertIsNone(d["fx"])
        self.assertIsNone(d["fy"])
        self.assertIsNone(d["cx"])
        self.assertIsNone(d["cy"])
        self.assertIsNone(d["distortion_coefficients"])

    def test_pipeline_calibration_propagation_end_to_end(self):
        """Verify pipeline preserves calibrated camera parameters into visual_metadata and top-level metadata."""
        calib_file = str(self.temp_path / "calib.json")
        with open(calib_file, "w", encoding="utf-8") as f:
            json.dump({"fx": 900.0, "fy": 900.0, "cx": 640.0, "cy": 360.0}, f)

        config = S1Config(
            video_path=self.video_path,
            calibration_path=calib_file,
            output_dir=str(self.temp_path / "calib_pipe"),
            sampling_mode="fixed",
            sampling_interval=5,
        )
        pipeline = S1Pipeline(config=config)
        s1_out = pipeline.run()

        self.assertEqual(s1_out.status, "completed")
        vis_meta_calib = s1_out.visual_observations.visual_metadata["camera_calibration"]
        self.assertTrue(vis_meta_calib["is_calibrated"])
        self.assertEqual(vis_meta_calib["fx"], 900.0)
        self.assertEqual(vis_meta_calib["cx"], 640.0)


if __name__ == "__main__":
    unittest.main()

