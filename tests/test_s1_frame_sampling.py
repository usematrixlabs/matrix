"""Unit tests for S1 Fixed-Interval Frame Sampling (Phase 4).

Verifies sequential frame decoding, configurable sampling intervals,
zero-padded disk saving (frame_000001.jpg), capture timestamp generation,
sub-range clipping, and memory-efficient streaming.
"""

import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from visual_perception import S1Config, S1Output
from visual_perception._internal.types import Frame
from visual_perception._internal.frame_extractor import FrameExtractor
from visual_perception._internal.pipeline import S1Pipeline


def create_sample_video(
    file_path: str,
    width: int = 640,
    height: int = 480,
    fps: float = 30.0,
    num_frames: int = 60,
) -> str:
    """Helper to synthesize a test UAV video with distinct colored frames."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
    for i in range(num_frames):
        # Generate distinctive frame content
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = [(i * 4) % 255, (i * 8) % 255, 200]
        # Add frame index text
        cv2.putText(frame, f"F_{i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        out.write(frame)
    out.release()
    return file_path


class TestS1FrameSampling(unittest.TestCase):
    """Tests for sequential fixed-interval frame sampling."""

    def setUp(self):
        """Setup temporary test environment and 60-frame 30 FPS synthetic video."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.frames_dir = self.temp_path / "frames"

        self.video_path = str(self.temp_path / "uav_flight_60frames.mp4")
        create_sample_video(self.video_path, width=640, height=480, fps=30.0, num_frames=60)

    def tearDown(self):
        """Cleanup temporary files."""
        self.temp_dir.cleanup()

    def test_fixed_interval_sampling_step10(self):
        """Verify that extracting 60 frames @ 30 FPS with interval=10 extracts exactly 6 frames."""
        extractor = FrameExtractor(
            video_path=self.video_path,
            sampling_mode="fixed",
            sampling_interval=10,
        )
        frames = extractor.extract(output_dir=str(self.frames_dir))

        # 60 frames with step 10 -> frames 0, 10, 20, 30, 40, 50 (6 frames)
        self.assertEqual(len(frames), 6)

        # Check frame IDs (zero-padded 6 digits)
        expected_ids = [f"frame_{i:06d}" for i in range(1, 7)]
        self.assertEqual([f.frame_id for f in frames], expected_ids)

        # Check timestamps (0.0, 10/30, 20/30, 30/30, 40/30, 50/30)
        expected_timestamps = [0.0, 10.0 / 30.0, 20.0 / 30.0, 30.0 / 30.0, 40.0 / 30.0, 50.0 / 30.0]
        for frame, exp_t in zip(frames, expected_timestamps):
            self.assertAlmostEqual(frame.timestamp, exp_t, places=4)

        # Check that files exist on disk and can be read
        for frame in frames:
            self.assertTrue(os.path.exists(frame.image_path))
            loaded = cv2.imread(frame.image_path)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.shape[1], 640)
            self.assertEqual(loaded.shape[0], 480)

    def test_fps_sampling_mode(self):
        """Verify fps sampling mode: 30 FPS video at 5.0 FPS target extracts 10 frames."""
        extractor = FrameExtractor(
            video_path=self.video_path,
            sampling_mode="fps",
            frame_rate=5.0,  # step = 30 / 5 = 6
        )
        frames = extractor.extract(output_dir=str(self.frames_dir))

        # 60 frames with step 6 -> 10 frames
        self.assertEqual(len(frames), 10)
        self.assertEqual(frames[0].frame_id, "frame_000001")
        self.assertEqual(frames[-1].frame_id, "frame_000010")

    def test_all_frames_sampling_mode(self):
        """Verify sampling_mode='all' extracts every single frame."""
        small_video = str(self.temp_path / "small.mp4")
        create_sample_video(small_video, width=320, height=240, fps=10.0, num_frames=15)

        extractor = FrameExtractor(
            video_path=small_video,
            sampling_mode="all",
        )
        frames = extractor.extract(output_dir=str(self.frames_dir))
        self.assertEqual(len(frames), 15)

    def test_time_range_clipping(self):
        """Verify extracting only between start_time and end_time."""
        extractor = FrameExtractor(
            video_path=self.video_path,
            sampling_mode="fixed",
            sampling_interval=5,
        )
        # Extract between 0.5s (frame 15) and 1.5s (frame 45) -> 31 candidate frames / 5 = 7 frames
        frames = extractor.extract(
            start_time=0.5,
            end_time=1.5,
            output_dir=str(self.frames_dir),
        )

        self.assertGreater(len(frames), 0)
        self.assertGreaterEqual(frames[0].timestamp, 0.49)
        self.assertLessEqual(frames[-1].timestamp, 1.51)

    def test_custom_resolution_and_format(self):
        """Verify custom resizing and PNG image format."""
        config = S1Config(
            video_path=self.video_path,
            target_width=320,
            target_height=240,
            image_format="png",
            sampling_mode="fixed",
            sampling_interval=20,
        )
        extractor = FrameExtractor(config=config)
        frames = extractor.extract(output_dir=str(self.frames_dir))

        self.assertEqual(len(frames), 3)
        for frame in frames:
            self.assertTrue(frame.image_path.endswith(".png"))
            self.assertEqual(frame.image_width, 320)
            self.assertEqual(frame.image_height, 240)
            img = cv2.imread(frame.image_path)
            self.assertEqual(img.shape[:2], (240, 320))

    def test_pipeline_end_to_end_frame_sampling(self):
        """Verify S1Pipeline extracts frames, populates visual observations, and records metadata."""
        out_dir = str(self.temp_path / "pipeline_run")
        config = S1Config(
            video_path=self.video_path,
            output_dir=out_dir,
            sampling_mode="fixed",
            sampling_interval=15,
            log_level="DEBUG",
        )
        pipeline = S1Pipeline(config=config)
        result = pipeline.run()

        self.assertIsInstance(result, S1Output)
        self.assertEqual(result.status, "completed")
        self.assertEqual(len(result.visual_observations.frames), 4)  # 60 frames / 15 = 4 frames
        self.assertEqual(len(result.visual_observations.frame_ordering), 4)
        self.assertEqual(result.visual_observations.visual_metadata["total_frames_extracted"], 4)
        self.assertEqual(result.visual_observations.visual_metadata["sampling_interval"], 15)


if __name__ == "__main__":
    unittest.main()

