"""S1 Performance Evaluation & Benchmarking Suite (Phase 13).

Measures processing runtime, throughput (FPS), memory footprint (peak RAM),
disk storage consumption, and keyframe computational overhead across pipeline configurations:
1. Sampling Only
2. Sampling + Quality Assessment
3. Sampling + Quality Assessment + Keyframe Detection
"""

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import tracemalloc
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np

from .config import S1Config
from .logger import get_logger
from .pipeline import S1Pipeline
from .types import S1Output


@dataclass
class BenchmarkRunResult:
    """Represents performance metrics collected for a single pipeline execution mode."""

    mode_name: str
    description: str
    total_time_seconds: float
    throughput_fps: float
    total_frames_processed: int
    observations_generated: int
    keyframes_generated: int
    output_storage_kb: float
    peak_ram_mb: float
    overhead_vs_baseline_percent: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize benchmark run result to dictionary."""
        return asdict(self)


class S1BenchmarkRunner:
    """Runs automated benchmarks and comparative performance evaluations for S1."""

    def __init__(self, log_level: str = "INFO"):
        """Initialize the benchmark runner.

        Parameters:
            log_level (str): Logging level.
        """
        self.logger = get_logger(self.__class__.__name__, log_level=log_level)

    @staticmethod
    def _calculate_directory_size_kb(dir_path: Union[str, Path]) -> float:
        """Calculate total disk footprint of a directory in Kilobytes."""
        p = Path(dir_path)
        if not p.exists():
            return 0.0
        total_bytes = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        return round(total_bytes / 1024.0, 2)

    def benchmark_mode(
        self,
        video_path: str,
        mode_name: str,
        config_overrides: Dict[str, Any],
        output_dir: Optional[str] = None,
    ) -> BenchmarkRunResult:
        """Benchmark a single pipeline configuration mode.

        Parameters:
            video_path (str): Path to input video file.
            mode_name (str): Identifier for this mode.
            config_overrides (Dict[str, Any]): Configuration properties to override.
            output_dir (Optional[str]): Temporary output destination.

        Returns:
            BenchmarkRunResult: Collected performance metrics.
        """
        temp_dir: Optional[tempfile.TemporaryDirectory] = None
        if output_dir is None:
            temp_dir = tempfile.TemporaryDirectory()
            target_out = temp_dir.name
        else:
            target_out = output_dir

        try:
            cfg = S1Config(
                video_path=video_path,
                output_dir=target_out,
                log_level="WARNING",
            )
            for k, v in config_overrides.items():
                setattr(cfg, k, v)

            pipeline = S1Pipeline(config=cfg)

            # Start memory and time tracking
            tracemalloc.start()
            start_time = time.perf_counter()

            s1_out: S1Output = pipeline.run(video_path=video_path, output_dir=target_out)

            elapsed = max(0.0001, time.perf_counter() - start_time)
            current_mem, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Compute metrics
            frames_count = len(s1_out.visual_observations.frames)
            keyframes_count = len(s1_out.visual_observations.keyframes)
            video_frames = (
                s1_out.metadata.get("video_metadata", {}).get("frame_count", frames_count)
                if s1_out.metadata
                else frames_count
            )

            throughput = round(video_frames / elapsed, 2)
            storage_kb = self._calculate_directory_size_kb(target_out)
            peak_ram_mb = round(peak_mem / (1024.0 * 1024.0), 3)

            return BenchmarkRunResult(
                mode_name=mode_name,
                description=config_overrides.get("description", mode_name),
                total_time_seconds=round(elapsed, 4),
                throughput_fps=throughput,
                total_frames_processed=video_frames,
                observations_generated=frames_count,
                keyframes_generated=keyframes_count,
                output_storage_kb=storage_kb,
                peak_ram_mb=peak_ram_mb,
            )
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()

    def run_comparative_benchmark(
        self,
        video_path: str,
        sampling_interval: int = 5,
    ) -> Dict[str, Any]:
        """Execute the standard 3-mode comparative benchmark required by Phase 13.

        Compares:
        1. Mode 1: Sampling Only (Minimal baseline)
        2. Mode 2: Sampling + Visual Quality Assessment
        3. Mode 3: Sampling + Quality Assessment + Content-Change Keyframing

        Parameters:
            video_path (str): Path to input video file.
            sampling_interval (int): Frame sampling interval.

        Returns:
            Dict[str, Any]: Structured comparative benchmark results.
        """
        self.logger.info("Starting comparative S1 performance benchmark on '%s'...", video_path)

        modes = [
            (
                "Sampling Only",
                "Fixed-interval frame extraction without quality scoring or keyframing",
                {
                    "sampling_mode": "fixed",
                    "sampling_interval": sampling_interval,
                    "enable_quality_assessment": False,
                    "keyframe_method": "uniform",
                    "description": "Sampling Only",
                },
            ),
            (
                "Sampling + Quality",
                "Frame extraction with Laplacian blur, exposure, and texture quality assessment",
                {
                    "sampling_mode": "fixed",
                    "sampling_interval": sampling_interval,
                    "enable_quality_assessment": True,
                    "keyframe_method": "uniform",
                    "description": "Sampling + Quality Assessment",
                },
            ),
            (
                "Sampling + Quality + Keyframes",
                "Full pipeline with quality assessment and Bhattacharyya histogram keyframing",
                {
                    "sampling_mode": "fixed",
                    "sampling_interval": sampling_interval,
                    "enable_quality_assessment": True,
                    "keyframe_method": "content_change",
                    "description": "Full Pipeline (Quality + Keyframes)",
                },
            ),
        ]

        results: List[BenchmarkRunResult] = []
        baseline_time: Optional[float] = None

        for name, desc, overrides in modes:
            self.logger.info("Executing benchmark mode: '%s'...", name)
            res = self.benchmark_mode(video_path=video_path, mode_name=name, config_overrides=overrides)

            if baseline_time is None:
                baseline_time = res.total_time_seconds
                res.overhead_vs_baseline_percent = 0.0
            else:
                overhead = ((res.total_time_seconds - baseline_time) / max(0.0001, baseline_time)) * 100.0
                res.overhead_vs_baseline_percent = round(overhead, 2)

            results.append(res)

        # Compute keyframe overhead specifically
        time_mode2 = results[1].total_time_seconds
        time_mode3 = results[2].total_time_seconds
        keyframe_overhead_percent = round(((time_mode3 - time_mode2) / max(0.0001, time_mode2)) * 100.0, 2)

        report = {
            "video_source": video_path,
            "sampling_interval": sampling_interval,
            "baseline_mode": "Sampling Only",
            "keyframe_overhead_percent": keyframe_overhead_percent,
            "results": [r.to_dict() for r in results],
            "markdown_table": self.format_markdown_table(results),
        }

        self.logger.info("Benchmark complete. Keyframe computational overhead: %.2f%%", keyframe_overhead_percent)
        return report

    @staticmethod
    def format_markdown_table(results: List[BenchmarkRunResult]) -> str:
        """Format benchmark results into a clean GitHub Markdown table."""
        lines = [
            "| Mode | Total Time (s) | Throughput (FPS) | Observations | Keyframes | Peak RAM (MB) | Storage (KB) | Overhead vs Baseline |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        for r in results:
            overhead_str = f"+{r.overhead_vs_baseline_percent:.1f}%" if r.overhead_vs_baseline_percent > 0 else "0.0% (Baseline)"
            lines.append(
                f"| **{r.mode_name}** | {r.total_time_seconds:.4f}s | {r.throughput_fps:.1f} FPS | {r.observations_generated} | {r.keyframes_generated} | {r.peak_ram_mb:.3f} MB | {r.output_storage_kb:.1f} KB | {overhead_str} |"
            )
        return "\n".join(lines)


def generate_synthetic_benchmark_video(file_path: str, num_frames: int = 60, width: int = 640, height: int = 480) -> str:
    """Synthesize a video with geometric movement and textures for benchmarking."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(file_path, fourcc, 30.0, (width, height))
    for i in range(num_frames):
        frame = np.full((height, width, 3), (i * 4) % 255, dtype=np.uint8)
        # Add dynamic textures and geometric circles
        cv2.rectangle(frame, (50, 50), (200, 200), (0, 255, 0), -1)
        cv2.circle(frame, (100 + (i * 5) % 400, 200 + (i * 3) % 200), 30, (0, 0, 255), -1)
        out.write(frame)
    out.release()
    return file_path


def main() -> None:
    """CLI entrypoint for running S1 performance benchmarks."""
    parser = argparse.ArgumentParser(description="Matrix S1 Performance Benchmark (Phase 13)")
    parser.add_argument("--video", "-v", type=str, help="Path to video file (if omitted, synthesizes test video)")
    parser.add_argument("--interval", "-i", type=int, default=5, help="Sampling interval")
    parser.add_argument("--json", action="store_true", help="Output results as raw JSON")

    args = parser.parse_args()

    runner = S1BenchmarkRunner()

    temp_video_dir: Optional[tempfile.TemporaryDirectory] = None
    if args.video and os.path.exists(args.video):
        target_video = args.video
    else:
        temp_video_dir = tempfile.TemporaryDirectory()
        target_video = str(Path(temp_video_dir.name) / "benchmark_flight.mp4")
        generate_synthetic_benchmark_video(target_video, num_frames=60)
        print(f"[INFO] Generated synthetic 60-frame benchmark video at: {target_video}\n")

    try:
        report = runner.run_comparative_benchmark(video_path=target_video, sampling_interval=args.interval)

        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print("\n=== Matrix S1 Performance Benchmark Report ===")
            print(f"Video Source: {report['video_source']}")
            print(f"Sampling Interval: Every {report['sampling_interval']} frames\n")
            print(report["markdown_table"])
            print(f"\nKeyframe Computational Overhead: {report['keyframe_overhead_percent']}%\n")
    finally:
        if temp_video_dir is not None:
            temp_video_dir.cleanup()


if __name__ == "__main__":
    main()

