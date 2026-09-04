"""Unit tests for S1 Visual Quality Assessment (Phase 7).

Verifies calculation of quality metrics for blur (Laplacian variance),
exposure (underexposure, overexposure), low-feature content, and corruption.
Ensures non-destructive reporting where poor observations are tagged rather than dropped.
"""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from visual_perception import S1Config, S1Output
from visual_perception._internal.types import Frame, QualityAssessment
from visual_perception._internal.frame_extractor import FrameExtractor
from visual_perception._internal.quality_assessor import QualityAssessor
from visual_perception._internal.pipeline import S1Pipeline


def create_textured_sharp_image(width: int = 320, height: int = 240) -> np.ndarray:
    """Generate a high-contrast sharp image with rich texture, gradients, and corners."""
    np.random.seed(42)
    # Background gradient
    x = np.linspace(50, 200, width, dtype=np.uint8)
    y = np.linspace(50, 200, height, dtype=np.uint8)
    xx, yy = np.meshgrid(x, y)
    base = ((xx.astype(np.float32) + yy.astype(np.float32)) / 2).astype(np.uint8)
    img = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

    # Add dense high-contrast corner elements
    for cx in range(20, width - 20, 25):
        for cy in range(20, height - 20, 25):
            cv2.rectangle(img, (cx - 8, cy - 8), (cx + 8, cy + 8), (0, 0, 255), -1)
            cv2.circle(img, (cx, cy), 4, (255, 255, 0), -1)
            cv2.drawMarker(img, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 10, 2)

    return img


class TestS1QualityAssessment(unittest.TestCase):
    """Tests for QualityAssessor and quality metadata tracking."""

    def setUp(self):
        """Setup assessor instance and temporary directory."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.assessor = QualityAssessor(
            blur_threshold=100.0,
            underexposure_threshold=30.0,
            overexposure_threshold=230.0,
            low_feature_threshold=50,
            min_entropy_threshold=3.5,
        )

    def tearDown(self):
        """Cleanup temporary files."""
        self.temp_dir.cleanup()

    def test_good_quality_frame(self):
        """Verify sharp, well-exposed, textured frame evaluates to GOOD."""
        sharp_img = create_textured_sharp_image()
        assessment = self.assessor.assess_frame(sharp_img, frame_id="frame_000001")

        self.assertIsInstance(assessment, QualityAssessment)
        self.assertEqual(assessment.status, "GOOD")
        self.assertGreater(assessment.blur_score, 100.0)
        self.assertGreaterEqual(assessment.exposure_mean, 30.0)
        self.assertLessEqual(assessment.exposure_mean, 230.0)
        self.assertGreater(assessment.feature_count, 50)
        self.assertFalse(assessment.is_corrupted)
        self.assertGreaterEqual(assessment.quality_score, 70.0)
        self.assertEqual(len(assessment.flags), 0)

    def test_blurry_frame_detection(self):
        """Verify synthetically blurred frame evaluates to BLURRY."""
        sharp_img = create_textured_sharp_image()
        # Apply intense Gaussian blur
        blurry_img = cv2.GaussianBlur(sharp_img, (31, 31), sigmaX=15.0)
        assessment = self.assessor.assess_frame(blurry_img, frame_id="frame_000002")

        self.assertIn("BLURRY", assessment.flags)
        self.assertEqual(assessment.status, "BLURRY")
        self.assertLess(assessment.blur_score, 100.0)

    def test_overexposed_frame_detection(self):
        """Verify washed-out high-brightness frame evaluates to OVEREXPOSED."""
        overexposed_img = np.full((240, 320, 3), 245, dtype=np.uint8)
        assessment = self.assessor.assess_frame(overexposed_img, frame_id="frame_000003")

        self.assertIn("OVEREXPOSED", assessment.flags)
        self.assertEqual(assessment.status, "OVEREXPOSED")
        self.assertGreater(assessment.exposure_mean, 230.0)

    def test_underexposed_frame_detection(self):
        """Verify dark low-brightness frame evaluates to UNDEREXPOSED."""
        underexposed_img = np.full((240, 320, 3), 15, dtype=np.uint8)
        assessment = self.assessor.assess_frame(underexposed_img, frame_id="frame_000004")

        self.assertIn("UNDEREXPOSED", assessment.flags)
        self.assertEqual(assessment.status, "UNDEREXPOSED")
        self.assertLess(assessment.exposure_mean, 30.0)

    def test_low_feature_frame_detection(self):
        """Verify solid flat-colored frame evaluates to LOW_FEATURE."""
        flat_img = np.full((240, 320, 3), 128, dtype=np.uint8)
        assessment = self.assessor.assess_frame(flat_img, frame_id="frame_000005")

        self.assertIn("LOW_FEATURE", assessment.flags)
        self.assertEqual(assessment.status, "LOW_FEATURE")
        self.assertEqual(assessment.feature_count, 0)

    def test_corrupted_frame_detection(self):
        """Verify None, empty array, or NaN array evaluates to CORRUPTED."""
        # Empty array
        empty_arr = np.array([], dtype=np.uint8)
        assessment_empty = self.assessor.assess_frame(empty_arr)
        self.assertEqual(assessment_empty.status, "CORRUPTED")
        self.assertTrue(assessment_empty.is_corrupted)
        self.assertEqual(assessment_empty.quality_score, 0.0)

        # None
        assessment_none = self.assessor.assess_frame(None)
        self.assertEqual(assessment_none.status, "CORRUPTED")

        # NaN array
        nan_img = np.full((10, 10, 3), np.nan)
        assessment_nan = self.assessor.assess_frame(nan_img)
        self.assertEqual(assessment_nan.status, "CORRUPTED")

    def test_non_destructive_reporting_in_frame_extractor(self):
        """Verify that poor frames are tagged with quality records without being dropped."""
        video_path = str(self.temp_path / "mixed_quality.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(video_path, fourcc, 10.0, (320, 240))

        # Write 5 sharp frames
        sharp = create_textured_sharp_image(320, 240)
        for _ in range(5):
            out.write(sharp)
        # Write 5 blurry frames
        blurred = cv2.GaussianBlur(sharp, (35, 35), 15.0)
        for _ in range(5):
            out.write(blurred)
        # Write 5 overexposed frames
        bright = np.full((240, 320, 3), 245, dtype=np.uint8)
        for _ in range(5):
            out.write(bright)
        out.release()

        extractor = FrameExtractor(video_path=video_path, sampling_mode="all")
        frames = extractor.extract(output_dir=str(self.temp_path / "frames"))

        # Exactly all 15 frames must be retained
        self.assertEqual(len(frames), 15)

        # Verify quality records exist on all frames
        for f in frames:
            self.assertIsNotNone(f.quality)
            self.assertIsInstance(f.quality, QualityAssessment)

        # Verify statuses
        statuses = [f.quality.status for f in frames]
        self.assertIn("GOOD", statuses)
        self.assertIn("BLURRY", statuses)
        self.assertIn("OVEREXPOSED", statuses)

    def test_pipeline_quality_summary_metadata(self):
        """Verify S1Pipeline aggregates quality status counts in visual_metadata."""
        video_path = str(self.temp_path / "pipeline_quality.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(video_path, fourcc, 10.0, (320, 240))

        sharp = create_textured_sharp_image(320, 240)
        for _ in range(6):
            out.write(sharp)
        out.release()

        config = S1Config(
            video_path=video_path,
            output_dir=str(self.temp_path / "out"),
            sampling_mode="all",
        )
        pipeline = S1Pipeline(config=config)
        s1_out = pipeline.run()

        self.assertEqual(s1_out.status, "completed")
        quality_summary = s1_out.visual_observations.visual_metadata["quality_summary"]

        self.assertIn("GOOD", quality_summary)
        self.assertEqual(quality_summary["GOOD"], 6)
        self.assertEqual(quality_summary["BLURRY"], 0)


if __name__ == "__main__":
    unittest.main()

