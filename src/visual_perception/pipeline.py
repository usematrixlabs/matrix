"""S1 Visual Perception Pipeline Runner.

Coordinates video decoding, frame extraction, keyframe selection,
and metadata preservation conforming to the S1 -> S2 interface contract.
"""

import argparse
import json
import os
import sys
import time
from typing import Optional

from .config import S1Config
from .exceptions import VideoValidationError
from .frame_extractor import FrameExtractor
from .keyframe_selector import KeyframeSelector
from .logger import get_logger
from .metadata_extractor import MetadataExtractor
from .types import (
    Frame,
    Keyframe,
    S1Output,
    UAVTelemetry,
    VideoMetadata,
    VideoMetadataRecord,
    VisualObservations,
)
from .video_validator import VideoValidator


class S1Pipeline:
    """End-to-end pipeline runner for Subsystem 1 (Visual Perception)."""

    def __init__(self, config: Optional[S1Config] = None):
        """Initialize the S1 pipeline runner.

        Parameters:
            config (Optional[S1Config]): Subsystem configuration.
        """
        self.config = config or S1Config()
        self.logger = get_logger("S1Pipeline", log_level=self.config.log_level, log_file=self.config.log_file)
        self.validator = VideoValidator(log_level=self.config.log_level)
        self.metadata_extractor = MetadataExtractor(log_level=self.config.log_level)
        self.extractor = FrameExtractor(config=self.config)
        self.selector = KeyframeSelector(config=self.config)

    def run(
        self,
        video_path: Optional[str] = None,
        telemetry_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        strict_validation: bool = True,
    ) -> S1Output:
        """Execute the Visual Perception pipeline.

        Parameters:
            video_path (Optional[str]): Path to the UAV input video.
            telemetry_path (Optional[str]): Path to optional UAV telemetry file.
            output_dir (Optional[str]): Destination directory for outputs.
            strict_validation (bool): If True, raises validation exceptions immediately.

        Returns:
            S1Output: Structured output compliant with S1 -> S2 interface contract.
        """
        start_time = time.time()
        self.logger.info("Initializing S1 Visual Perception pipeline...")

        # Override config paths if provided directly
        if video_path:
            self.config.video_path = video_path
            self.extractor.video_path = video_path
        if telemetry_path:
            self.config.telemetry_path = telemetry_path
        if output_dir:
            self.config.output_dir = output_dir
            self.config.frames_dir = str(os.path.join(output_dir, "frames"))
            self.config.keyframes_dir = str(os.path.join(output_dir, "keyframes"))

        # Ensure output directories are created
        self.config.ensure_directories()

        # Step 1: Ingest telemetry if available (without interpreting it)
        telemetry_data = UAVTelemetry()
        if self.config.telemetry_path and os.path.exists(self.config.telemetry_path):
            self.logger.info("Ingesting UAV telemetry from '%s'", self.config.telemetry_path)
            try:
                with open(self.config.telemetry_path, "r", encoding="utf-8") as f:
                    raw_telemetry = json.load(f)
                telemetry_data = UAVTelemetry(**{k: v for k, v in raw_telemetry.items() if k in UAVTelemetry.__dataclass_fields__})
            except Exception as e:
                self.logger.warning("Could not parse telemetry file '%s': %s", self.config.telemetry_path, e)
        elif self.config.telemetry_path:
            self.logger.warning("Telemetry file '%s' not found.", self.config.telemetry_path)

        # Step 2: Validate UAV Video Input & Extract Structured Metadata (Phase 2 & Phase 3)
        video_metadata_record: Optional[VideoMetadataRecord] = None
        if self.config.video_path:
            self.logger.info("Extracting structured metadata for '%s'...", self.config.video_path)
            try:
                video_metadata_record = self.metadata_extractor.extract(
                    video_path=self.config.video_path,
                    sidecar_path=self.config.telemetry_path,
                    start_time_offset=self.config.time_start,
                )
                self.extractor.video_metadata = video_metadata_record.video
                self.extractor.metadata_record = video_metadata_record
            except VideoValidationError as e:
                self.logger.error("Video metadata extraction failed: %s", e)
                if strict_validation:
                    raise
                # Return failure contract
                return S1Output(
                    status="failed",
                    metadata={
                        "subsystem": "S1_Visual_Perception",
                        "error": str(e),
                        "video_source": self.config.video_path,
                    },
                )
        else:
            self.logger.info("No video path provided to pipeline. Proceeding in initial/empty mode.")

        # Step 3: Extract & Sample Frames (Phase 4)
        self.logger.info("Running frame extraction...")
        extracted_frames = self.extractor.extract(
            start_time=self.config.time_start,
            end_time=self.config.time_end,
            sampling_interval=self.config.sampling_interval,
            sampling_mode=self.config.sampling_mode,
            output_dir=self.config.frames_dir,
        )

        # Step 4: Select Keyframes
        self.logger.info("Running keyframe selection...")
        selected_keyframes = self.selector.select(frames=extracted_frames)

        # Step 5: Assemble S1 -> S2 Interface Output
        frame_ordering = [f.frame_id for f in extracted_frames]
        visual_obs = VisualObservations(
            frames=extracted_frames,
            keyframes=selected_keyframes,
            frame_ordering=frame_ordering,
            visual_metadata={
                "total_frames_extracted": len(extracted_frames),
                "total_keyframes_selected": len(selected_keyframes),
                "sampling_mode": self.config.sampling_mode,
                "sampling_interval": self.config.sampling_interval,
                "extraction_fps": self.config.frame_rate,
                "keyframe_method": self.config.keyframe_method,
            },
        )

        elapsed = time.time() - start_time
        temporal_info = {
            "processing_time_seconds": round(elapsed, 4),
            "start_time_offset": self.config.time_start,
            "end_time_offset": self.config.time_end,
        }
        if video_metadata_record:
            temporal_info["frame_interval_seconds"] = video_metadata_record.timing.frame_interval_seconds
            temporal_info["video_duration_seconds"] = video_metadata_record.timing.duration_seconds

        s1_output = S1Output(
            visual_observations=visual_obs,
            temporal_information=temporal_info,
            available_uav_information=telemetry_data,
            status="completed",
            metadata={
                "subsystem": "S1_Visual_Perception",
                "version": "0.1.0",
                "video_source": self.config.video_path,
                "video_metadata": video_metadata_record.video.to_dict() if video_metadata_record else None,
                "video_metadata_record": video_metadata_record.to_dict() if video_metadata_record else None,
                "output_dir": self.config.output_dir,
            },
        )

        self.logger.info(
            "S1 Pipeline completed in %.3fs (Extracted %d frames, %d keyframes)",
            elapsed,
            len(extracted_frames),
            len(selected_keyframes),
        )
        return s1_output


def main() -> None:
    """CLI entrypoint for running S1 Visual Perception."""
    parser = argparse.ArgumentParser(description="Matrix S1 — Visual Perception Pipeline")
    parser.add_argument("--video", "-v", type=str, help="Path to input UAV video file")
    parser.add_argument("--telemetry", "-t", type=str, help="Path to optional telemetry JSON file")
    parser.add_argument("--config", "-c", type=str, help="Path to YAML configuration file")
    parser.add_argument("--output-dir", "-o", type=str, help="Output directory for frames & observations")
    parser.add_argument("--sampling-mode", "-m", type=str, choices=["fixed", "fps", "all"], help="Sampling mode")
    parser.add_argument("--sampling-interval", "-i", type=int, help="Fixed interval in frames (e.g. 10)")
    parser.add_argument("--frame-rate", "-r", type=float, help="Extraction rate (FPS)")
    parser.add_argument("--image-format", type=str, choices=["jpg", "jpeg", "png"], help="Frame image format")
    parser.add_argument("--log-level", "-l", type=str, default="INFO", help="Logging level")
    parser.add_argument("--save-output", "-s", type=str, help="Path to save output JSON contract")

    args = parser.parse_args()

    # Load configuration
    if args.config:
        config = S1Config.from_yaml(args.config)
    else:
        config = S1Config()

    if args.video:
        config.video_path = args.video
    if args.telemetry:
        config.telemetry_path = args.telemetry
    if args.output_dir:
        config.output_dir = args.output_dir
        config.frames_dir = str(os.path.join(args.output_dir, "frames"))
        config.keyframes_dir = str(os.path.join(args.output_dir, "keyframes"))
    if args.sampling_mode:
        config.sampling_mode = args.sampling_mode
    if args.sampling_interval:
        config.sampling_interval = args.sampling_interval
    if args.frame_rate:
        config.frame_rate = args.frame_rate
    if args.image_format:
        config.image_format = args.image_format
    if args.log_level:
        config.log_level = args.log_level

    pipeline = S1Pipeline(config=config)
    try:
        result = pipeline.run()
    except VideoValidationError as e:
        print(f"\n[ERROR] S1 Video Validation Failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Output result as formatted JSON to stdout or file
    result_dict = result.to_dict()
    if args.save_output:
        with open(args.save_output, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2)
        print(f"S1 Output saved to: {args.save_output}")
    else:
        print("\n--- S1 Output Summary ---")
        print(json.dumps(result_dict, indent=2))


if __name__ == "__main__":
    main()
