"""Unit tests for S1 UAV Video Input & Validation (Phase 2).

Verifies video existence checks, container validation, decoder open check,
readability, metadata extraction (FPS, frame count, resolution, duration, codec),
and error reporting for missing, empty, corrupt, and unsupported files.
"""

import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from visual_perception import S1Config, S1Output
from visual_perception._internal.frame_extractor import FrameExtractor
from visual_perception._internal.pipeline import S1Pipeline
from visual_perception._internal.exceptions import VideoCorruptError, VideoFormatError, VideoMetadataError, VideoNotFoundError, VideoUnreadableError, VideoValidationError
from visual_perception._internal.types import VideoMetadata
from visual_perception._internal.video_validator import VideoValidator


def create_synthetic_video(
    file_path: str,
    width: int = 320,
    height: int = 240,
    fps: float = 10.0,
    num_frames: int = 20,
) -> str:
    """Create a small valid MP4 test video for testing."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
    for i in range(num_frames):
        # Create a frame with solid color and simple gradient
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = [(i * 10) % 255, (i * 20) % 255, 150]
        out.write(frame)
    out.release()
    return file_path


class TestS1VideoValidation(unittest.TestCase):
    """Tests for VideoValidator and validation integration across S1."""

    def setUp(self):
        """Create temporary test directory and synthetic video."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.validator = VideoValidator(log_level="DEBUG")

        # Create a valid synthetic video
        self.valid_video_path = str(self.temp_path / "uav_flight_test.mp4")
        create_synthetic_video(
            self.valid_video_path,
            width=320,
            height=240,
            fps=10.0,
            num_frames=20,
        )

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_valid_video_metadata_extraction(self):
        """Verify that a valid video is accepted and all metadata fields are accurately extracted."""
        metadata = self.validator.validate(self.valid_video_path)

        self.assertIsInstance(metadata, VideoMetadata)
        self.assertTrue(metadata.is_valid)
        self.assertEqual(metadata.filename, "uav_flight_test.mp4")
        self.assertEqual(metadata.width, 320)
        self.assertEqual(metadata.height, 240)
        self.assertAlmostEqual(metadata.fps, 10.0, places=1)
        self.assertEqual(metadata.frame_count, 20)
        self.assertAlmostEqual(metadata.duration_seconds, 2.0, places=1)
        self.assertGreater(metadata.file_size_bytes, 0)
        self.assertTrue(len(metadata.validation_notes) >= 4)

        # Test dictionary serialization
        meta_dict = metadata.to_dict()
        self.assertEqual(meta_dict["width"], 320)
        self.assertEqual(meta_dict["height"], 240)

    def test_missing_video_raises_not_found(self):
        """Verify that a non-existent video path raises VideoNotFoundError."""
        missing_path = str(self.temp_path / "does_not_exist.mp4")
        with self.assertRaises(VideoNotFoundError) as ctx:
            self.validator.validate(missing_path)
        self.assertIn("Video file not found", str(ctx.exception))

    def test_empty_0byte_file_raises_corrupt(self):
        """Verify that a 0-byte video file raises VideoCorruptError."""
        empty_path = str(self.temp_path / "empty_video.mp4")
        with open(empty_path, "wb") as f:
            pass  # Create 0-byte file

        with self.assertRaises(VideoCorruptError) as ctx:
            self.validator.validate(empty_path)
        self.assertIn("0 bytes", str(ctx.exception))

    def test_corrupt_header_raises_corrupt_or_unreadable(self):
        """Verify that a file with garbage bytes raises VideoCorruptError or VideoUnreadableError."""
        corrupt_path = str(self.temp_path / "corrupt_header.mp4")
        with open(corrupt_path, "wb") as f:
            f.write(b"NOT_A_VALID_MP4_HEADER_GARBAGE_BYTES_1234567890" * 50)

        with self.assertRaises((VideoCorruptError, VideoUnreadableError)) as ctx:
            self.validator.validate(corrupt_path)
        self.assertTrue(isinstance(ctx.exception, VideoValidationError))

    def test_unsupported_format_raises_format_error(self):
        """Verify that unsupported file extensions raise VideoFormatError."""
        txt_path = str(self.temp_path / "flight_log.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("Some drone flight notes...")

        with self.assertRaises(VideoFormatError) as ctx:
            self.validator.validate(txt_path)
        self.assertIn("Unsupported video container", str(ctx.exception))

    def test_frame_extractor_integration(self):
        """Verify FrameExtractor validates and caches video metadata."""
        extractor = FrameExtractor(video_path=self.valid_video_path)
        meta = extractor.validate()

        self.assertIsNotNone(extractor.video_metadata)
        self.assertEqual(meta.width, 320)
        self.assertEqual(meta.height, 240)
        self.assertEqual(extractor.video_metadata.frame_count, 20)

    def test_pipeline_integration_valid_video(self):
        """Verify S1Pipeline validates video and embeds VideoMetadata in S1Output."""
        config = S1Config(
            video_path=self.valid_video_path,
            output_dir=str(self.temp_path / "pipeline_out"),
            log_level="DEBUG",
        )
        pipeline = S1Pipeline(config=config)
        result = pipeline.run()

        self.assertIsInstance(result, S1Output)
        self.assertEqual(result.status, "completed")
        self.assertIn("video_metadata", result.metadata)
        self.assertIsNotNone(result.metadata["video_metadata"])
        self.assertEqual(result.metadata["video_metadata"]["width"], 320)
        self.assertEqual(result.metadata["video_metadata"]["height"], 240)

    def test_pipeline_integration_missing_video(self):
        """Verify S1Pipeline raises or reports error on invalid/missing input."""
        config = S1Config(
            video_path=str(self.temp_path / "missing.mp4"),
            output_dir=str(self.temp_path / "pipeline_out"),
            log_level="DEBUG",
        )
        pipeline = S1Pipeline(config=config)

        # Strict validation should raise exception
        with self.assertRaises(VideoNotFoundError):
            pipeline.run(strict_validation=True)

        # Non-strict validation should return failed status
        failed_result = pipeline.run(strict_validation=False)
        self.assertEqual(failed_result.status, "failed")
        self.assertIn("error", failed_result.metadata)


if __name__ == "__main__":
    unittest.main()

