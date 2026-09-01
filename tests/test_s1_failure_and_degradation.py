"""Unit tests for S1 Failure & Degradation Handling (Phase 11).

Verifies explicit distinction between hard failures (missing/corrupt/unsupported video),
graceful degradation (insufficient observations, high blur ratio), and non-blocking
missing optional metadata (calibration, telemetry) with structured diagnostics.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.visual_perception import (
    ObservationPackager,
    QualityAssessor,
    S1Config,
    S1DiagnosticsEvaluator,
    S1Output,
    S1Pipeline,
    VideoCorruptError,
    VideoFormatError,
    VideoNotFoundError,
    VideoValidationError,
)


def create_solid_color_video(file_path: str, width: int = 320, height: int = 240, fps: float = 10.0, num_frames: int = 10, color=(128, 128, 128)) -> str:
    """Helper to synthesize a video with uniform flat color (inducing blur/low features)."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
    for _ in range(num_frames):
        frame = np.full((height, width, 3), color, dtype=np.uint8)
        out.write(frame)
    out.release()
    return file_path


def create_textured_video(file_path: str, width: int = 320, height: int = 240, fps: float = 10.0, num_frames: int = 20) -> str:
    """Helper to synthesize a rich textured video."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
        cv2.circle(frame, (100 + i * 2, 100), 20, (0, 255, 0), -1)
        out.write(frame)
    out.release()
    return file_path


class TestS1FailureAndDegradation(unittest.TestCase):
    """Tests for Phase 11 Failure and Degradation handling."""

    def setUp(self):
        """Setup temporary test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        """Cleanup temporary directory."""
        self.temp_dir.cleanup()

    def test_missing_video_hard_failure(self):
        """Verify non-existent video path raises VideoNotFoundError or returns status=failed."""
        missing_path = str(self.temp_path / "non_existent_flight.mp4")
        pipeline = S1Pipeline()

        # Strict validation raises exception
        with self.assertRaises(VideoNotFoundError):
            pipeline.run(video_path=missing_path, strict_validation=True)

        # Non-strict validation returns structured failed output
        out = pipeline.run(video_path=missing_path, strict_validation=False)
        self.assertEqual(out.status, "failed")
        self.assertGreater(len(out.errors), 0)
        self.assertIn("not found", out.errors[0].lower())
        self.assertEqual(out.diagnostics.get("health_status"), "failed")

    def test_corrupt_video_hard_failure(self):
        """Verify garbage video file raises VideoCorruptError or returns status=failed."""
        corrupt_file = str(self.temp_path / "corrupt_flight.mp4")
        with open(corrupt_file, "wb") as f:
            f.write(b"GARBAGE_HEADER_DATA_1234567890")

        pipeline = S1Pipeline()
        with self.assertRaises(VideoValidationError):
            pipeline.run(video_path=corrupt_file, strict_validation=True)

        out = pipeline.run(video_path=corrupt_file, strict_validation=False)
        self.assertEqual(out.status, "failed")
        self.assertGreater(len(out.errors), 0)

    def test_unsupported_video_format_hard_failure(self):
        """Verify unsupported video format raises VideoFormatError."""
        bad_format_file = str(self.temp_path / "flight.unsupported_codec")
        with open(bad_format_file, "w") as f:
            f.write("text content")

        pipeline = S1Pipeline()
        with self.assertRaises(VideoFormatError):
            pipeline.run(video_path=bad_format_file, strict_validation=True)

    def test_corrupt_frame_flagged_and_rejected(self):
        """Verify corrupt frame is classified as CORRUPTED and never marked as a valid keyframe."""
        assessor = QualityAssessor()
        corrupt_img = np.zeros((0, 0, 3), dtype=np.uint8)  # Empty array
        q = assessor.assess(corrupt_img)

        self.assertEqual(q.status, "CORRUPTED")
        self.assertTrue(q.is_corrupted)
        self.assertEqual(q.quality_score, 0.0)

    def test_missing_optional_calibration_continues_with_warning(self):
        """Verify missing camera calibration produces warning and status=completed without halting."""
        video_path = str(self.temp_path / "test_normal.mp4")
        create_textured_video(video_path, num_frames=20)

        config = S1Config(
            video_path=video_path,
            output_dir=str(self.temp_path / "out_no_calib"),
            calibration_path=None,  # explicitly missing
            sampling_mode="fixed",
            sampling_interval=2,
            min_valid_observations=5,
        )
        pipeline = S1Pipeline(config=config)
        out = pipeline.run()

        self.assertEqual(out.status, "completed")
        self.assertTrue(any("missing_camera_calibration" in w for w in out.warnings))
        self.assertFalse(out.diagnostics["sensor_availability"]["camera_calibration"])
        self.assertGreater(len(out.visual_observations.frames), 0)

    def test_missing_optional_telemetry_continues_with_warning(self):
        """Verify missing telemetry produces warning and status=completed."""
        video_path = str(self.temp_path / "test_normal_2.mp4")
        create_textured_video(video_path, num_frames=20)

        config = S1Config(
            video_path=video_path,
            output_dir=str(self.temp_path / "out_no_telem"),
            telemetry_path=None,  # explicitly missing
            sampling_mode="fixed",
            sampling_interval=2,
            min_valid_observations=5,
        )
        pipeline = S1Pipeline(config=config)
        out = pipeline.run()

        self.assertEqual(out.status, "completed")
        self.assertTrue(any("missing_uav_telemetry" in w for w in out.warnings))
        self.assertFalse(out.diagnostics["sensor_availability"]["telemetry_present"])

    def test_insufficient_observations_produces_degraded_status(self):
        """Verify pipeline returns status=degraded when valid observations are below threshold."""
        video_path = str(self.temp_path / "test_short.mp4")
        create_textured_video(video_path, num_frames=10)

        # Sampling step 8 from 10 frames yields only 2 observations (< threshold 5)
        config = S1Config(
            video_path=video_path,
            output_dir=str(self.temp_path / "out_short"),
            sampling_mode="fixed",
            sampling_interval=8,
            min_valid_observations=5,
        )
        pipeline = S1Pipeline(config=config)
        out = pipeline.run()

        self.assertEqual(out.status, "degraded")
        self.assertTrue(out.diagnostics["is_degraded"])
        self.assertTrue(any("insufficient_valid_observations" in w for w in out.warnings))
        self.assertEqual(out.diagnostics["observations_summary"]["valid_count"], 2)

    def test_high_blur_ratio_produces_degraded_status(self):
        """Verify pipeline returns status=degraded when observations are predominantly blurry."""
        video_path = str(self.temp_path / "test_blurred.mp4")
        create_solid_color_video(video_path, num_frames=20, color=(120, 120, 120))

        config = S1Config(
            video_path=video_path,
            output_dir=str(self.temp_path / "out_blur"),
            sampling_mode="fixed",
            sampling_interval=2,
            min_valid_observations=5,
            max_degraded_ratio=0.5,
        )
        pipeline = S1Pipeline(config=config)
        out = pipeline.run()

        self.assertEqual(out.status, "degraded")
        self.assertTrue(out.diagnostics["is_degraded"])
        self.assertTrue(any("high_visual_degradation_ratio" in w or "insufficient" in w for w in out.warnings))

    def test_diagnostics_evaluator_standalone(self):
        """Verify S1DiagnosticsEvaluator directly."""
        evaluator = S1DiagnosticsEvaluator()
        status, warnings, diag = evaluator.evaluate_health(
            frames=[],
            keyframes=[],
            video_record=None,
            telemetry_loaded=False,
        )
        self.assertEqual(status, "completed")
        self.assertIn("observations_summary", diag)


if __name__ == "__main__":
    unittest.main()

