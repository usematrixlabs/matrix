"""Unit tests for S1 Video Metadata Extraction (Phase 3).

Verifies extraction of stream geometry, FPS, frame counts, per-frame timing calculations,
optional camera/flight/sensor metadata parsing, and explicit None representation.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.visual_perception import S1Config, S1Output
from src.visual_perception._internal.types import CameraMetadata, FlightMetadata, FrameTimingInfo, SensorMetadata, VideoMetadataRecord
from src.visual_perception._internal.frame_extractor import FrameExtractor
from src.visual_perception._internal.metadata_extractor import MetadataExtractor
from src.visual_perception._internal.pipeline import S1Pipeline


def create_test_video(file_path: str, width: int = 640, height: int = 480, fps: float = 30.0, num_frames: int = 60) -> str:
    """Helper to synthesize a test video."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.full((height, width, 3), fill_value=(i * 4) % 256, dtype=np.uint8)
        out.write(frame)
    out.release()
    return file_path


class TestS1MetadataExtraction(unittest.TestCase):
    """Tests for MetadataExtractor and structured video metadata representation."""

    def setUp(self):
        """Setup temporary test environment and sample video."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.extractor = MetadataExtractor(log_level="DEBUG")

        self.video_path = str(self.temp_path / "uav_flight_test.mp4")
        create_test_video(self.video_path, width=640, height=480, fps=30.0, num_frames=60)

    def tearDown(self):
        """Cleanup temporary files."""
        self.temp_dir.cleanup()

    def test_stream_metadata_extraction(self):
        """Verify stream dimensions, FPS, frame count, and duration are accurately extracted."""
        record = self.extractor.extract(self.video_path)

        self.assertIsInstance(record, VideoMetadataRecord)
        self.assertEqual(record.video.width, 640)
        self.assertEqual(record.video.height, 480)
        self.assertAlmostEqual(record.video.fps, 30.0, places=1)
        self.assertEqual(record.video.frame_count, 60)
        self.assertAlmostEqual(record.video.duration_seconds, 2.0, places=1)

    def test_frame_timing_info_calculations(self):
        """Verify per-frame timestamp indexing and frame interval calculations."""
        record = self.extractor.extract(self.video_path, start_time_offset=10.0)
        timing = record.timing

        self.assertIsInstance(timing, FrameTimingInfo)
        self.assertAlmostEqual(timing.fps, 30.0, places=1)
        self.assertAlmostEqual(timing.frame_interval_seconds, 1.0 / 30.0, places=4)
        self.assertEqual(timing.total_frames, 60)
        self.assertEqual(timing.start_timestamp, 10.0)
        self.assertAlmostEqual(timing.end_timestamp, 12.0, places=1)

        # Test per-frame timestamp calculation
        # Frame 0 should be start_timestamp
        self.assertAlmostEqual(timing.get_timestamp_for_frame(0), 10.0, places=4)
        # Frame 30 (1 second in) should be 11.0s
        self.assertAlmostEqual(timing.get_timestamp_for_frame(30), 11.0, places=4)
        # Frame 60 should be 12.0s
        self.assertAlmostEqual(timing.get_timestamp_for_frame(60), 12.0, places=4)

        # Test timestamp-to-index lookup
        self.assertEqual(timing.get_frame_index_for_timestamp(10.0), 0)
        self.assertEqual(timing.get_frame_index_for_timestamp(11.0), 30)
        self.assertEqual(timing.get_frame_index_for_timestamp(10.5), 15)

        # Test error handling on negative index
        with self.assertRaises(ValueError):
            timing.get_timestamp_for_frame(-1)

    def test_explicit_none_for_missing_optional_metadata(self):
        """Verify that when no auxiliary metadata is provided, optional fields are explicitly None."""
        record = self.extractor.extract(self.video_path)

        self.assertIsNone(record.camera.camera_make)
        self.assertIsNone(record.camera.camera_model)
        self.assertIsNone(record.camera.focal_length_mm)
        self.assertIsNone(record.camera.field_of_view_deg)

        self.assertIsNone(record.flight.flight_id)
        self.assertIsNone(record.flight.aircraft_model)
        self.assertIsNone(record.flight.takeoff_timestamp)

        self.assertFalse(record.sensor.has_gps)
        self.assertFalse(record.sensor.has_imu)
        self.assertIsNone(record.sensor.gps_sampling_rate_hz)

    def test_sidecar_metadata_ingestion(self):
        """Verify that optional camera, flight, and sensor metadata is parsed when provided."""
        sidecar_data = {
            "camera": {
                "camera_make": "DJI",
                "camera_model": "Zenmuse P1",
                "focal_length_mm": 35.0,
                "sensor_width_mm": 35.9,
                "sensor_height_mm": 24.0,
                "field_of_view_deg": 63.5,
                "exposure_mode": "manual",
            },
            "flight": {
                "flight_id": "SIH26158_FLIGHT_01",
                "aircraft_model": "DJI Matrice 300 RTK",
                "takeoff_timestamp": 1725148800.0,
                "pilot_operator": "NTRO Operator 1",
                "mission_type": "reconnaissance_mapping",
            },
            "sensor": {
                "has_gps": True,
                "has_imu": True,
                "has_rtk": True,
                "gps_sampling_rate_hz": 10.0,
                "imu_sampling_rate_hz": 200.0,
                "coordinate_system": "WGS84",
                "altitude_reference": "MSL",
            },
        }

        # Test extraction via sidecar dictionary
        record = self.extractor.extract(self.video_path, sidecar_data=sidecar_data)

        self.assertEqual(record.camera.camera_make, "DJI")
        self.assertEqual(record.camera.camera_model, "Zenmuse P1")
        self.assertEqual(record.camera.focal_length_mm, 35.0)

        self.assertEqual(record.flight.flight_id, "SIH26158_FLIGHT_01")
        self.assertEqual(record.flight.aircraft_model, "DJI Matrice 300 RTK")

        self.assertTrue(record.sensor.has_gps)
        self.assertTrue(record.sensor.has_imu)
        self.assertTrue(record.sensor.has_rtk)
        self.assertEqual(record.sensor.gps_sampling_rate_hz, 10.0)
        self.assertEqual(record.sensor.coordinate_system, "WGS84")

        # Test extraction via sidecar JSON file
        sidecar_file = str(self.temp_path / "flight_telemetry.json")
        with open(sidecar_file, "w", encoding="utf-8") as f:
            json.dump(sidecar_data, f)

        record_from_file = self.extractor.extract(self.video_path, sidecar_path=sidecar_file)
        self.assertEqual(record_from_file.camera.camera_model, "Zenmuse P1")
        self.assertEqual(record_from_file.flight.flight_id, "SIH26158_FLIGHT_01")

    def test_deliverable_json_format(self):
        """Verify that record.to_dict() matches the deliverable specification."""
        record = self.extractor.extract(self.video_path)
        output_dict = record.to_dict()

        self.assertIn("video", output_dict)
        self.assertIn("timing", output_dict)
        self.assertIn("camera", output_dict)
        self.assertIn("flight", output_dict)
        self.assertIn("sensor", output_dict)

        video_section = output_dict["video"]
        self.assertEqual(video_section["width"], 640)
        self.assertEqual(video_section["height"], 480)
        self.assertAlmostEqual(video_section["fps"], 30.0, places=1)
        self.assertEqual(video_section["frame_count"], 60)
        self.assertAlmostEqual(video_section["duration_sec"], 2.0, places=1)

    def test_pipeline_integration(self):
        """Verify that S1Pipeline attaches structured video_metadata_record in S1Output."""
        config = S1Config(
            video_path=self.video_path,
            output_dir=str(self.temp_path / "out"),
        )
        pipeline = S1Pipeline(config=config)
        s1_out = pipeline.run()

        self.assertEqual(s1_out.status, "completed")
        self.assertIn("video_metadata_record", s1_out.metadata)
        record_dict = s1_out.metadata["video_metadata_record"]
        self.assertEqual(record_dict["video"]["width"], 640)
        self.assertIn("timing", record_dict)


if __name__ == "__main__":
    unittest.main()
