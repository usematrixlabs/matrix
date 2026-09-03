"""Level 4 Integration & Downstream Validation Tests (Phase 12 & Phase 14).

Verifies end-to-end integration and data contract compliance between
Subsystem 1 (Visual Perception), Subsystem 2 (Localization & Sensor Fusion),
and Subsystem 3 (3D Reconstruction).
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from visual_perception import S1Config
from visual_perception._internal.downstream_validator import DownstreamValidator
from visual_perception._internal.packager import ObservationPackager
from visual_perception._internal.pipeline import S1Pipeline


def create_integration_video(file_path: str, width: int = 640, height: int = 480, fps: float = 24.0, num_frames: int = 48) -> str:
    """Helper to synthesize a video simulating UAV flight motion with moving geometric structures."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.full((height, width, 3), (40 + i * 2) % 255, dtype=np.uint8)
        # Add simulated building structure
        cv2.rectangle(frame, (150 + i * 2, 100), (350 + i * 2, 350), (0, 180, 0), -1)
        # Add high-contrast corner features
        cv2.circle(frame, (200 + i * 2, 150), 15, (255, 255, 0), -1)
        cv2.circle(frame, (300 + i * 2, 300), 15, (0, 255, 255), -1)
        out.write(frame)
    out.release()
    return file_path


class TestS1DownstreamIntegration(unittest.TestCase):
    """Integration test suite for S1 -> S2 and S1 -> S3 consumer compatibility."""

    def setUp(self):
        """Setup temporary test environment and synthetic video."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        self.video_path = str(self.temp_path / "uav_integration_flight.mp4")
        create_integration_video(self.video_path, width=640, height=480, fps=24.0, num_frames=48)
        self.out_dir = str(self.temp_path / "s1_output")

        # Create camera calibration
        self.calib_path = str(self.temp_path / "camera_calib.json")
        calib_data = {
            "width": 640,
            "height": 480,
            "fx": 600.0,
            "fy": 600.0,
            "cx": 320.0,
            "cy": 240.0,
            "distortion_coefficients": [-0.1, 0.05, 0.0, 0.0, 0.0],
            "distortion_model": "radtan",
        }
        with open(self.calib_path, "w", encoding="utf-8") as f:
            json.dump(calib_data, f)

    def tearDown(self):
        """Cleanup temporary files."""
        self.temp_dir.cleanup()

    def test_s2_localization_handoff_and_trajectory_association(self):
        """Verify S2 Localization can consume observations, associate poses, and verify monotonic IDs."""
        config = S1Config(
            video_path=self.video_path,
            calibration_path=self.calib_path,
            output_dir=self.out_dir,
            sampling_mode="fixed",
            sampling_interval=4,
            keyframe_method="content_change",
            keyframe_change_threshold=0.08,
        )
        pipeline = S1Pipeline(config=config)
        s1_out = pipeline.run()

        self.assertEqual(s1_out.status, "completed")

        validator = DownstreamValidator()
        s2_report = validator.validate_s2_compatibility(self.out_dir)

        self.assertEqual(s2_report["status"], "compatible")
        self.assertEqual(s2_report["consumer"], "S2_Localization_and_Sensor_Fusion")
        self.assertEqual(s2_report["total_observations_ingested"], 12)  # 48 frames / 4 = 12
        self.assertGreater(s2_report["keyframes_identified"], 0)
        self.assertEqual(s2_report["trajectory_poses_mapped"], 12)
        self.assertTrue(s2_report["id_association_verified"])
        self.assertTrue(s2_report["timestamp_monotonicity_verified"])

    def test_s2_keyframe_vs_all_observations_query_flexibility(self):
        """Verify downstream S2 can seamlessly select all candidate observations or keyframes only."""
        config = S1Config(
            video_path=self.video_path,
            output_dir=self.out_dir,
            sampling_mode="all",
            keyframe_method="uniform",
            min_keyframe_interval_frames=4,
        )
        pipeline = S1Pipeline(config=config)
        pipeline.run()

        package = ObservationPackager.load_package(self.out_dir)
        all_obs = package["observations"]
        keyframe_obs = [obs for obs in all_obs if obs["keyframe"]]

        # S2 Tracking uses all 48 observations
        self.assertEqual(len(all_obs), 48)
        # S2 Pose Graph uses subset of keyframes
        self.assertGreater(len(keyframe_obs), 0)
        self.assertLess(len(keyframe_obs), len(all_obs))

        # Confirm non-keyframes are not deleted from disk
        root_dir = Path(self.out_dir)
        for obs in all_obs:
            self.assertTrue((root_dir / obs["image"]).exists())

    def test_s3_reconstruction_handoff_and_image_decodability(self):
        """Verify S3 3D Reconstruction can decode all observation images and read camera parameters."""
        config = S1Config(
            video_path=self.video_path,
            calibration_path=self.calib_path,
            output_dir=self.out_dir,
            sampling_mode="fixed",
            sampling_interval=6,
        )
        pipeline = S1Pipeline(config=config)
        pipeline.run()

        validator = DownstreamValidator()
        s3_report = validator.validate_s3_compatibility(self.out_dir)

        self.assertEqual(s3_report["status"], "compatible")
        self.assertEqual(s3_report["consumer"], "S3_3D_Reconstruction")
        self.assertEqual(s3_report["total_images_decoded"], 8)  # 48 / 6 = 8
        self.assertEqual(s3_report["image_dimensions"], "640x480")
        self.assertTrue(s3_report["is_calibrated"])
        self.assertTrue(s3_report["intrinsics_available"])
        self.assertTrue(s3_report["multiview_ingestion_ready"])

    def test_missing_calibration_gracefully_consumed_by_s2_and_s3(self):
        """Verify downstream consumers can process uncalibrated observation packages."""
        config = S1Config(
            video_path=self.video_path,
            calibration_path=None,  # No calibration
            output_dir=self.out_dir,
            sampling_mode="fixed",
            sampling_interval=6,
        )
        pipeline = S1Pipeline(config=config)
        pipeline.run()

        validator = DownstreamValidator()
        s2_report = validator.validate_s2_compatibility(self.out_dir)
        s3_report = validator.validate_s3_compatibility(self.out_dir)

        self.assertEqual(s2_report["status"], "compatible")
        self.assertEqual(s3_report["status"], "compatible")
        self.assertFalse(s3_report["is_calibrated"])


if __name__ == "__main__":
    unittest.main()

