"""Unit tests for S1 Visual Perception Phase 1 Setup.

Verifies module imports, configuration management, data types conforming
to S1 -> S2 contract, logging, and pipeline execution.
"""

import json
import os
import tempfile
import unittest

from src.visual_perception import S1Config, S1Output
from src.visual_perception._internal.types import Frame, Keyframe, UAVTelemetry, VisualObservations
from src.visual_perception._internal.frame_extractor import FrameExtractor
from src.visual_perception._internal.keyframe_selector import KeyframeSelector
from src.visual_perception._internal.pipeline import S1Pipeline
from src.visual_perception._internal.logger import get_logger


class TestS1Setup(unittest.TestCase):
    """Tests for S1 module setup and initial execution."""

    def test_imports(self):
        """Verify all S1 exports can be imported successfully."""
        self.assertIsNotNone(Frame)
        self.assertIsNotNone(Keyframe)
        self.assertIsNotNone(UAVTelemetry)
        self.assertIsNotNone(VisualObservations)
        self.assertIsNotNone(S1Output)
        self.assertIsNotNone(S1Config)
        self.assertIsNotNone(S1Pipeline)
        self.assertIsNotNone(FrameExtractor)
        self.assertIsNotNone(KeyframeSelector)
        self.assertIsNotNone(get_logger)

    def test_types_contract_serialization(self):
        """Verify data models conform to the S1 -> S2 interface contract."""
        frame = Frame(
            frame_id="frame_0001",
            timestamp=0.5,
            image_path="data/output/frames/frame_0001.png",
            image_width=1920,
            image_height=1080,
        )
        self.assertEqual(frame.frame_id, "frame_0001")
        self.assertEqual(frame.to_dict()["image_width"], 1920)

        keyframe = Keyframe(
            frame_id="frame_0001",
            timestamp=0.5,
            image_path="data/output/keyframes/frame_0001.png",
            score=92.5,
        )
        self.assertEqual(keyframe.score, 92.5)

        telemetry = UAVTelemetry(
            gps_coordinates={"lat": 12.9716, "lon": 77.5946, "alt": 920.0},
            gnss_status="fixed",
        )
        self.assertEqual(telemetry.to_dict()["gnss_status"], "fixed")

        s1_output = S1Output(
            visual_observations=VisualObservations(
                frames=[frame],
                keyframes=[keyframe],
                frame_ordering=[frame.frame_id],
            ),
            available_uav_information=telemetry,
            status="completed",
        )
        serialized = s1_output.to_dict()
        self.assertEqual(serialized["status"], "completed")
        self.assertEqual(len(serialized["visual_observations"]["frames"]), 1)
        self.assertEqual(serialized["available_uav_information"]["gnss_status"], "fixed")

    def test_config_management(self):
        """Verify config initialization, dictionary conversion, and file loading."""
        config = S1Config(
            frame_rate=3.0,
            keyframe_method="laplacian_variance",
            quality_threshold=65.0,
        )
        self.assertEqual(config.frame_rate, 3.0)
        self.assertEqual(config.keyframe_method, "laplacian_variance")

        # Test from_dict
        cfg_dict = {"frame_rate": 5.0, "quality_threshold": 80.0, "custom_setting": 123}
        loaded_cfg = S1Config.from_dict(cfg_dict)
        self.assertEqual(loaded_cfg.frame_rate, 5.0)
        self.assertEqual(loaded_cfg.quality_threshold, 80.0)
        self.assertEqual(loaded_cfg.extra_params.get("custom_setting"), 123)

        # Test YAML saving and loading
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            config.save_yaml(tmp_path)
            loaded_from_yaml = S1Config.from_yaml(tmp_path)
            self.assertEqual(loaded_from_yaml.frame_rate, config.frame_rate)
            self.assertEqual(loaded_from_yaml.keyframe_method, config.keyframe_method)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_logger(self):
        """Verify logger instantiation."""
        logger = get_logger("TestLogger", log_level="DEBUG")
        self.assertEqual(logger.name, "TestLogger")

    def test_pipeline_execution(self):
        """Verify initial S1 pipeline execution produces valid S1Output contract."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = S1Config(
                output_dir=temp_dir,
                frame_rate=2.0,
                log_level="DEBUG",
            )
            pipeline = S1Pipeline(config=config)
            result = pipeline.run()

            self.assertIsInstance(result, S1Output)
            self.assertEqual(result.status, "completed")
            self.assertIn("total_frames_extracted", result.visual_observations.visual_metadata)
            self.assertTrue(os.path.exists(temp_dir))


if __name__ == "__main__":
    unittest.main()

