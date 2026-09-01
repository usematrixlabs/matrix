"""Unit tests for S1 Keyframe Detection (Phase 8).

Verifies visual change-based and uniform keyframe selection, marking observations with
'is_keyframe: bool', non-destructive retention of all candidate frames, reproducible
selection, and keyframe density measurement.
"""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.visual_perception import (
    Frame,
    FrameExtractor,
    Keyframe,
    KeyframeSelector,
    QualityAssessment,
    S1Config,
    S1Output,
    S1Pipeline,
)


def create_dynamic_scene_video(file_path: str, width: int = 320, height: int = 240, fps: float = 10.0) -> str:
    """Helper to synthesize a video with 2 distinct scene transitions."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))

    # Scene 1: 10 frames of textured blue pattern
    for i in range(10):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = 200  # Blue
        cv2.circle(frame, (50 + i * 2, 50), 30, (0, 255, 255), -1)
        out.write(frame)

    # Scene 2: 10 frames of textured bright green pattern (major content change)
    for i in range(10):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 1] = 220  # Green
        cv2.rectangle(frame, (100, 100), (200, 200), (255, 0, 255), -1)
        out.write(frame)

    out.release()
    return file_path


class TestS1KeyframeDetection(unittest.TestCase):
    """Tests for KeyframeSelector and keyframe observation marking."""

    def setUp(self):
        """Setup temporary test environment and synthetic dynamic video."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        self.video_path = str(self.temp_path / "dynamic_scene.mp4")
        create_dynamic_scene_video(self.video_path, width=320, height=240, fps=10.0)

    def tearDown(self):
        """Cleanup temporary files."""
        self.temp_dir.cleanup()

    def test_keyframe_marking_and_non_keyframe_retention(self):
        """Verify that all observations are retained and marked with is_keyframe: bool."""
        config = S1Config(
            video_path=self.video_path,
            output_dir=str(self.temp_path / "out"),
            sampling_mode="all",
            keyframe_method="content_change",
            keyframe_change_threshold=0.10,
        )
        pipeline = S1Pipeline(config=config)
        s1_out = pipeline.run()

        frames = s1_out.visual_observations.frames
        keyframes = s1_out.visual_observations.keyframes

        # All 20 candidate frames must remain present
        self.assertEqual(len(frames), 20)

        # Every frame must have a boolean is_keyframe attribute
        for f in frames:
            self.assertIsInstance(f.is_keyframe, bool)

        # Count of frames where is_keyframe is True must match len(keyframes)
        keyframe_flags_count = sum(1 for f in frames if f.is_keyframe)
        self.assertEqual(keyframe_flags_count, len(keyframes))
        self.assertGreater(len(keyframes), 0)

        # Non-keyframes exist and remain available
        non_keyframes_count = sum(1 for f in frames if not f.is_keyframe)
        self.assertGreater(non_keyframes_count, 0)

    def test_keyframe_selection_reproducibility(self):
        """Verify that running keyframe detection twice produces identical results."""
        config = S1Config(
            video_path=self.video_path,
            output_dir=str(self.temp_path / "rep_run"),
            sampling_mode="all",
            keyframe_method="content_change",
        )
        extractor = FrameExtractor(config=config)
        frames_run1 = extractor.extract(output_dir=str(self.temp_path / "frames1"))
        selector = KeyframeSelector(config=config)
        kf1 = selector.select(frames=frames_run1)

        frames_run2 = extractor.extract(output_dir=str(self.temp_path / "frames2"))
        kf2 = selector.select(frames=frames_run2)

        self.assertEqual(len(kf1), len(kf2))
        for k1, k2 in zip(kf1, kf2):
            self.assertEqual(k1.frame_id, k2.frame_id)
            self.assertEqual(k1.timestamp, k2.timestamp)
            self.assertEqual(k1.selection_reason, k2.selection_reason)

        flags1 = [f.is_keyframe for f in frames_run1]
        flags2 = [f.is_keyframe for f in frames_run2]
        self.assertEqual(flags1, flags2)

    def test_visual_content_change_detection(self):
        """Verify that significant scene transitions trigger keyframe detection."""
        extractor = FrameExtractor(video_path=self.video_path, sampling_mode="all")
        frames = extractor.extract(output_dir=str(self.temp_path / "change_frames"))

        selector = KeyframeSelector(
            config=S1Config(
                keyframe_method="content_change",
                keyframe_change_threshold=0.10,
                min_keyframe_interval_frames=1,
            )
        )
        keyframes = selector.select(frames=frames)

        # First frame (frame_000001) must be a keyframe
        self.assertEqual(keyframes[0].frame_id, "frame_000001")

        # The scene change at frame 11 (index 10 in 0-indexed) must be detected as a keyframe
        keyframe_ids = {k.frame_id for k in keyframes}
        self.assertIn("frame_000011", keyframe_ids)

    def test_keyframe_density_calculation(self):
        """Verify keyframe density calculation and metadata documentation."""
        frames = [
            Frame(frame_id=f"frame_{i:06d}", timestamp=float(i), image_path=f"p{i}", image_width=100, image_height=100, is_keyframe=(i % 2 == 0))
            for i in range(10)
        ]
        keyframes = [
            Keyframe(frame_id=f.frame_id, timestamp=f.timestamp, image_path=f.image_path)
            for f in frames if f.is_keyframe
        ]

        density = KeyframeSelector.calculate_keyframe_density(frames, keyframes)
        # 5 keyframes / 10 frames = 0.5
        self.assertEqual(density, 0.5)

    def test_uniform_keyframe_selection(self):
        """Verify uniform keyframe strategy subsamples at regular step."""
        extractor = FrameExtractor(video_path=self.video_path, sampling_mode="all")
        frames = extractor.extract(output_dir=str(self.temp_path / "uni_frames"))

        selector = KeyframeSelector(
            config=S1Config(
                keyframe_method="uniform",
                min_keyframe_interval_frames=4,
            )
        )
        keyframes = selector.select(frames=frames)

        # 20 frames with step 4 -> indices 0, 4, 8, 12, 16 -> 5 keyframes
        self.assertEqual(len(keyframes), 5)
        self.assertEqual(keyframes[0].frame_id, "frame_000001")
        self.assertEqual(keyframes[1].frame_id, "frame_000005")
        self.assertEqual(keyframes[2].frame_id, "frame_000009")

    def test_pipeline_keyframe_density_metadata(self):
        """Verify S1Pipeline exports keyframe_density in visual_metadata."""
        config = S1Config(
            video_path=self.video_path,
            output_dir=str(self.temp_path / "density_pipe"),
            sampling_mode="all",
            keyframe_method="uniform",
            min_keyframe_interval_frames=5,
        )
        pipeline = S1Pipeline(config=config)
        s1_out = pipeline.run()

        metadata = s1_out.visual_observations.visual_metadata
        self.assertIn("keyframe_density", metadata)
        self.assertEqual(metadata["total_frames_extracted"], 20)
        self.assertEqual(metadata["total_keyframes_selected"], 4)
        self.assertEqual(metadata["keyframe_density"], 0.2)


if __name__ == "__main__":
    unittest.main()
