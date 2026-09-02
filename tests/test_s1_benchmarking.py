"""Unit tests for S1 Performance Benchmarking Suite (Phase 13).

Verifies that S1BenchmarkRunner accurately measures total execution time,
throughput (FPS), memory footprint, storage usage, and computes comparative overhead.
"""

import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.visual_perception._internal.benchmark import S1BenchmarkRunner


def create_bench_video(file_path: str, width: int = 320, height: int = 240, fps: float = 30.0, num_frames: int = 30) -> str:
    """Helper to synthesize a test video for benchmarking."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.full((height, width, 3), (i * 8) % 255, dtype=np.uint8)
        cv2.circle(frame, (100 + i * 2, 100), 20, (0, 255, 0), -1)
        out.write(frame)
    out.release()
    return file_path


class TestS1Benchmarking(unittest.TestCase):
    """Tests for S1BenchmarkRunner."""

    def setUp(self):
        """Setup temporary test environment and synthetic video."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        self.video_path = str(self.temp_path / "uav_bench_flight.mp4")
        create_bench_video(self.video_path, width=320, height=240, fps=30.0, num_frames=30)

    def tearDown(self):
        """Cleanup temporary files."""
        self.temp_dir.cleanup()

    def test_single_mode_benchmark_metrics(self):
        """Verify benchmark_mode captures all required metrics."""
        runner = S1BenchmarkRunner()
        result = runner.benchmark_mode(
            video_path=self.video_path,
            mode_name="Sampling Only",
            config_overrides={"sampling_mode": "fixed", "sampling_interval": 5, "enable_quality_assessment": False},
        )

        self.assertEqual(result.mode_name, "Sampling Only")
        self.assertGreater(result.total_time_seconds, 0.0)
        self.assertGreater(result.throughput_fps, 0.0)
        self.assertEqual(result.observations_generated, 6)  # 30 / 5 = 6
        self.assertGreater(result.output_storage_kb, 0.0)
        self.assertGreaterEqual(result.peak_ram_mb, 0.0)

    def test_comparative_3mode_benchmark(self):
        """Verify run_comparative_benchmark evaluates all 3 standard modes and formats output table."""
        runner = S1BenchmarkRunner()
        report = runner.run_comparative_benchmark(video_path=self.video_path, sampling_interval=5)

        self.assertIn("video_source", report)
        self.assertIn("sampling_interval", report)
        self.assertIn("results", report)
        self.assertEqual(len(report["results"]), 3)

        mode_names = [r["mode_name"] for r in report["results"]]
        self.assertIn("Sampling Only", mode_names)
        self.assertIn("Sampling + Quality", mode_names)
        self.assertIn("Sampling + Quality + Keyframes", mode_names)

        # Baseline mode has 0% overhead
        self.assertEqual(report["results"][0]["overhead_vs_baseline_percent"], 0.0)
        self.assertIn("markdown_table", report)
        self.assertIn("| **Sampling Only** |", report["markdown_table"])
        self.assertIsInstance(report["keyframe_overhead_percent"], float)


if __name__ == "__main__":
    unittest.main()

