"""Unit tests for S1 Observation Packaging (Phase 10).

Verifies canonical directory packaging (s1_output/frames/ + observations.json),
JSON schema conformity, relative image path validity, programmatic loading,
and simulation of downstream S2 consumption.
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
    S1Config,
    S1Output,
    S1Pipeline,
)


def create_pack_test_video(file_path: str, width: int = 640, height: int = 480, fps: float = 20.0, num_frames: int = 30) -> str:
    """Helper to synthesize a test video."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.full((height, width, 3), (i * 8) % 255, dtype=np.uint8)
        # Add visual feature
        cv2.circle(frame, (100 + i * 5, 100), 20, (0, 255, 0), -1)
        out.write(frame)
    out.release()
    return file_path


class TestS1ObservationPackaging(unittest.TestCase):
    """Tests for ObservationPackager and canonical s1_output generation."""

    def setUp(self):
        """Setup temporary test environment and synthetic video."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        self.video_path = str(self.temp_path / "uav_flight_packaging.mp4")
        create_pack_test_video(self.video_path, width=640, height=480, fps=20.0, num_frames=30)
        self.out_dir = str(self.temp_path / "s1_output")

    def tearDown(self):
        """Cleanup temporary files."""
        self.temp_dir.cleanup()

    def test_canonical_directory_structure_and_files(self):
        """Verify s1_output/ contains frames/ and observations.json."""
        config = S1Config(
            video_path=self.video_path,
            output_dir=self.out_dir,
            sampling_mode="fixed",
            sampling_interval=5,
        )
        pipeline = S1Pipeline(config=config)
        s1_out = pipeline.run()

        self.assertEqual(s1_out.status, "completed")
        out_path = Path(self.out_dir)

        # Check directory structure
        self.assertTrue(out_path.exists())
        self.assertTrue((out_path / "frames").exists())
        self.assertTrue((out_path / "observations.json").exists())

        # Verify frames directory has expected number of image files
        image_files = list((out_path / "frames").glob("*.jpg"))
        self.assertEqual(len(image_files), 6)  # 30 frames / 5 = 6

    def test_observation_item_schema_compliance(self):
        """Verify each observation entry conforms to the specified schema."""
        config = S1Config(
            video_path=self.video_path,
            output_dir=self.out_dir,
            sampling_mode="fixed",
            sampling_interval=5,
        )
        pipeline = S1Pipeline(config=config)
        pipeline.run()

        data = ObservationPackager.load_package(self.out_dir)

        # Verify top-level metadata
        self.assertEqual(data["subsystem"], "S1_Visual_Perception")
        self.assertEqual(data["schema_version"], "1.0.0")
        self.assertEqual(data["total_observations"], 6)
        self.assertIn("observations", data)

        # Verify observation items
        for obs in data["observations"]:
            self.assertIn("observation_id", obs)
            self.assertTrue(obs["observation_id"].startswith("frame_"))
            self.assertIn("timestamp", obs)
            self.assertIsInstance(obs["timestamp"], (int, float))
            self.assertIn("image", obs)
            self.assertTrue(obs["image"].startswith("frames/"))
            self.assertIn("camera", obs)
            self.assertEqual(obs["camera"]["width"], 640)
            self.assertEqual(obs["camera"]["height"], 480)
            self.assertIn("quality", obs)
            self.assertIn("status", obs["quality"])
            self.assertIn("keyframe", obs)
            self.assertIsInstance(obs["keyframe"], bool)

    def test_relative_image_paths_resolve_to_existing_files(self):
        """Verify all relative image paths point to valid existing files on disk."""
        config = S1Config(
            video_path=self.video_path,
            output_dir=self.out_dir,
            sampling_mode="fixed",
            sampling_interval=6,
        )
        pipeline = S1Pipeline(config=config)
        pipeline.run()

        data = ObservationPackager.load_package(self.out_dir)
        root_dir = Path(self.out_dir)

        for obs in data["observations"]:
            rel_path = obs["image"]
            full_path = root_dir / rel_path
            self.assertTrue(full_path.exists(), f"Image path '{rel_path}' does not resolve to a file")
            self.assertGreater(full_path.stat().st_size, 0)

    def test_package_validation_catches_missing_image_or_corruption(self):
        """Verify validate_package raises ValueError if an image is missing or invalid."""
        config = S1Config(
            video_path=self.video_path,
            output_dir=self.out_dir,
            sampling_mode="fixed",
            sampling_interval=10,
        )
        pipeline = S1Pipeline(config=config)
        pipeline.run()

        # Healthy package passes validation
        self.assertTrue(ObservationPackager.validate_package(self.out_dir))

        # Delete one image file and verify validation fails
        root_dir = Path(self.out_dir)
        first_img = list((root_dir / "frames").glob("*.jpg"))[0]
        os.remove(first_img)

        with self.assertRaises(ValueError) as ctx:
            ObservationPackager.validate_package(self.out_dir)
        self.assertIn("image file does not exist", str(ctx.exception))

    def test_s2_consumption_simulation(self):
        """Simulate Subsystem 2 reading all observations vs keyframes only."""
        config = S1Config(
            video_path=self.video_path,
            output_dir=self.out_dir,
            sampling_mode="all",
            keyframe_method="uniform",
            min_keyframe_interval_frames=5,
        )
        pipeline = S1Pipeline(config=config)
        pipeline.run()

        package = ObservationPackager.load_package(self.out_dir)
        all_observations = package["observations"]

        # S2 Mode A: Tracking with all candidate observations
        self.assertEqual(len(all_observations), 30)

        # S2 Mode B: Feature matching with keyframes only
        keyframe_observations = [obs for obs in all_observations if obs["keyframe"]]
        self.assertGreater(len(keyframe_observations), 0)
        self.assertLess(len(keyframe_observations), len(all_observations))

        # All keyframe images must exist
        root_dir = Path(self.out_dir)
        for kf_obs in keyframe_observations:
            self.assertTrue((root_dir / kf_obs["image"]).exists())

    def test_schema_definition(self):
        """Verify get_json_schema returns valid dictionary."""
        schema = ObservationPackager.get_json_schema()
        self.assertEqual(schema["title"], "MatrixS1ObservationsPackage")
        self.assertIn("observations", schema["properties"])


if __name__ == "__main__":
    unittest.main()

