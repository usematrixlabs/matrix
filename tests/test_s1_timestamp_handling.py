"""Unit tests for S1 Timestamp Handling (Phase 6).

Verifies capture timestamp calculation from source video streams, strict monotonicity
validation across sequential observations, unit documentation ('seconds'), and keyframe
capture timing preservation.
"""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.visual_perception import (
    Frame,
    FrameExtractor,
    KeyframeSelector,
    S1Config,
    S1Output,
    S1Pipeline,
    TimestampHandler,
)


def create_timing_test_video(file_path: str, width: int = 320, height: int = 240, fps: float = 30.0, num_frames: int = 90) -> str:
    """Helper to synthesize a 3-second 30 FPS test video."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.full((height, width, 3), fill_value=(i * 2) % 255, dtype=np.uint8)
        out.write(frame)
    out.release()
    return file_path


class TestS1TimestampHandling(unittest.TestCase):
    """Tests for TimestampHandler and timestamp preservation throughout S1."""

    def setUp(self):
        """Setup temporary test environment and synthetic video."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        self.video_path = str(self.temp_path / "uav_flight_timing.mp4")
        create_timing_test_video(self.video_path, width=320, height=240, fps=30.0, num_frames=90)

    def tearDown(self):
        """Cleanup temporary directory."""
        self.temp_dir.cleanup()

    def test_calculate_timestamp_accuracy(self):
        """Verify accurate calculation of capture timestamps in seconds."""
        fps = 30.0
        # Frame 0 -> 0.0s
        self.assertEqual(TimestampHandler.calculate_timestamp(0, fps=fps), 0.0)
        # Frame 15 -> 0.5s
        self.assertEqual(TimestampHandler.calculate_timestamp(15, fps=fps), 0.5)
        # Frame 30 -> 1.0s
        self.assertEqual(TimestampHandler.calculate_timestamp(30, fps=fps), 1.0)
        # Frame 45 -> 1.5s
        self.assertEqual(TimestampHandler.calculate_timestamp(45, fps=fps), 1.5)

        # Test with start offset (e.g. 10.0s)
        self.assertEqual(TimestampHandler.calculate_timestamp(30, fps=fps, start_offset=10.0), 11.0)

        # Test with absolute UTC base time
        base_epoch = 1725148800.0
        t_utc = TimestampHandler.calculate_timestamp(30, fps=fps, base_time=base_epoch)
        self.assertEqual(t_utc, 1725148801.0)

        # Test error conditions
        with self.assertRaises(ValueError):
            TimestampHandler.calculate_timestamp(-1, fps=30.0)
        with self.assertRaises(ValueError):
            TimestampHandler.calculate_timestamp(0, fps=0.0)

    def test_monotonicity_validation_success_and_failure(self):
        """Verify that strictly increasing timestamps pass, while non-monotonic timestamps fail."""
        f1 = Frame(frame_id="frame_000001", timestamp=0.0, image_path="p1", image_width=100, image_height=100)
        f2 = Frame(frame_id="frame_000002", timestamp=0.333333, image_path="p2", image_width=100, image_height=100)
        f3 = Frame(frame_id="frame_000003", timestamp=0.666667, image_path="p3", image_width=100, image_height=100)

        # Monotonic list passes
        self.assertTrue(TimestampHandler.validate_monotonicity([f1, f2, f3]))

        # Duplicate timestamp fails
        f_dup = Frame(frame_id="frame_000004", timestamp=0.666667, image_path="p4", image_width=100, image_height=100)
        with self.assertRaises(ValueError) as ctx:
            TimestampHandler.validate_monotonicity([f1, f2, f3, f_dup])
        self.assertIn("Non-monotonic timestamp detected", str(ctx.exception))

        # Decreasing timestamp fails
        f_backwards = Frame(frame_id="frame_000005", timestamp=0.5, image_path="p5", image_width=100, image_height=100)
        with self.assertRaises(ValueError) as ctx:
            TimestampHandler.validate_monotonicity([f1, f2, f3, f_backwards])
        self.assertIn("Non-monotonic timestamp detected", str(ctx.exception))

        # Negative timestamp fails
        f_neg = Frame(frame_id="frame_000006", timestamp=-1.0, image_path="p6", image_width=100, image_height=100)
        with self.assertRaises(ValueError) as ctx:
            TimestampHandler.validate_monotonicity([f_neg])
        self.assertIn("negative timestamp", str(ctx.exception))

    def test_observation_timing_dict_format(self):
        """Verify the standardized observation timing record format."""
        frame = Frame(frame_id="frame_000123", timestamp=12.34, image_path="p", image_width=100, image_height=100)
        timing_dict = TimestampHandler.to_observation_timing_dict(frame)

        self.assertEqual(timing_dict["observation_id"], "frame_000123")
        self.assertEqual(timing_dict["timestamp"], 12.34)
        self.assertEqual(timing_dict["unit"], "seconds")

    def test_extracted_frames_have_valid_monotonic_capture_times(self):
        """Verify extracted frames from real video stream carry monotonic capture timestamps."""
        extractor = FrameExtractor(
            video_path=self.video_path,
            sampling_mode="fixed",
            sampling_interval=10,
        )
        frames = extractor.extract(output_dir=str(self.temp_path / "frames"))

        # 90 frames / 10 = 9 frames
        self.assertEqual(len(frames), 9)

        # Verify all timestamps are positive and strictly monotonic
        timestamps = [f.timestamp for f in frames]
        for i in range(len(timestamps) - 1):
            self.assertLess(timestamps[i], timestamps[i + 1])

        # Verify exact timing matches capture frame indices (0, 10/30, 20/30...)
        for idx, f in enumerate(frames):
            expected_t = round((idx * 10) / 30.0, 6)
            self.assertAlmostEqual(f.timestamp, expected_t, places=4)

    def test_keyframe_capture_timestamp_preservation(self):
        """Verify that selected keyframes preserve parent observation capture timestamps."""
        extractor = FrameExtractor(video_path=self.video_path, sampling_mode="fixed", sampling_interval=5)
        frames = extractor.extract(output_dir=str(self.temp_path / "cand_frames"))

        selector = KeyframeSelector(config=S1Config(keyframe_method="uniform"))
        keyframes = selector.select(frames=frames)

        frame_time_map = {f.frame_id: f.timestamp for f in frames}
        for kf in keyframes:
            # Keyframe timestamp must match the source frame timestamp exactly
            self.assertEqual(kf.timestamp, frame_time_map[kf.frame_id])

    def test_pipeline_temporal_information_documentation(self):
        """Verify S1Pipeline documents time unit, monotonic status, and timing metadata."""
        config = S1Config(
            video_path=self.video_path,
            output_dir=str(self.temp_path / "pipeline_run"),
            sampling_mode="fixed",
            sampling_interval=15,
        )
        pipeline = S1Pipeline(config=config)
        s1_out = pipeline.run()

        self.assertEqual(s1_out.status, "completed")
        temp_info = s1_out.temporal_information

        # Verify documented timestamp unit
        self.assertEqual(temp_info["time_unit"], "seconds")
        self.assertEqual(temp_info["time_reference"], "relative_capture_time")
        self.assertTrue(temp_info["is_monotonic"])
        self.assertTrue(s1_out.visual_observations.visual_metadata["capture_timestamps_validated"])


if __name__ == "__main__":
    unittest.main()

