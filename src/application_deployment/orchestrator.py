"""
S5 — Application & Deployment: Pipeline Orchestrator

Orchestrates the full end-to-end Matrix pipeline (S1 -> S2 -> S3 -> S4 -> S5),
manages job lifecycles, aggregates execution metrics across all subsystems,
and packages deliverables for visualization and downstream delivery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union
import numpy as np

# S1 imports
from src.visual_perception.config import S1Config
from src.visual_perception.pipeline import S1Pipeline
from src.visual_perception.types import S1Output

# S2 imports
from src.localization_sensor_fusion.adapters.s1_adapter import S1InputAdapter
from src.localization_sensor_fusion.exporters.s2_exporter import S2Exporter
from src.localization_sensor_fusion.fusion.fusion_engine import SensorFusionEngine
from src.localization_sensor_fusion.schemas.contracts import (
    CameraPose as S2CameraPose,
    LocalizationMeta,
    LocalizationQuality,
    Position,
    QuaternionOrientation,
    S2ObservationOutput,
)

# S3 imports
from src.reconstruction.models.s3_output import S3ReconstructionResult
from src.reconstruction.models.schema import S2Payload
from src.reconstruction.pipeline import S3ReconstructionPipeline

# S4 imports
from src.georeferencing_validation.control_points import ControlPoints
from src.georeferencing_validation.crs import CoordinateReference
from src.georeferencing_validation.georeferencer import GeoreferencedResult, Georeferencer
from src.georeferencing_validation.input import ReconstructionInput


logger = logging.getLogger("MatrixOrchestrator")


class Orchestrator:
    """End-to-end pipeline orchestrator for Matrix UAV video-to-3D reconstruction."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the orchestrator with configuration settings.

        Parameters:
            config: Optional configuration dictionary containing settings for S1, S2, S3, S4, and S5.
        """
        self.config = dict(config or {})
        self.pipeline_status = "initialized"
        self.current_job_id: Optional[str] = None
        self.stages_status: Dict[str, Dict[str, Any]] = {
            "s1_perception": {"status": "pending", "duration_seconds": 0.0},
            "s2_localization": {"status": "pending", "duration_seconds": 0.0},
            "s3_reconstruction": {"status": "pending", "duration_seconds": 0.0},
            "s4_georeferencing": {"status": "pending", "duration_seconds": 0.0},
            "s5_presentation": {"status": "pending", "duration_seconds": 0.0},
        }
        self.last_result: Optional[Dict[str, Any]] = None

    def get_status(self) -> Dict[str, Any]:
        """Report the current pipeline processing status and stage metrics."""
        return {
            "status": self.pipeline_status,
            "job_id": self.current_job_id,
            "stages": dict(self.stages_status),
        }

    def run_pipeline(
        self,
        video_path: Optional[Union[str, Path]] = None,
        telemetry_path: Optional[Union[str, Path]] = None,
        calibration_path: Optional[Union[str, Path]] = None,
        gcp_data: Optional[Union[ControlPoints, Dict[str, Any], np.ndarray]] = None,
        source_crs: Optional[CoordinateReference] = None,
        target_crs: Optional[CoordinateReference] = None,
        output_dir: Optional[Union[str, Path]] = None,
        job_id: Optional[str] = None,
        s2_payload: Optional[Union[S2Payload, Dict[str, Any]]] = None,
        s3_result: Optional[S3ReconstructionResult] = None,
    ) -> Dict[str, Any]:
        """Execute the Matrix processing pipeline across all subsystems.

        Parameters:
            video_path: Path to UAV video file for S1.
            telemetry_path: Optional path to telemetry JSON file.
            calibration_path: Optional path to camera calibration file.
            gcp_data: Optional Ground Control Points for S4 georeferencing.
            source_crs: Optional source coordinate reference (defaults to local with world policy).
            target_crs: Optional target coordinate reference (defaults to UTM Zone 43N).
            output_dir: Root directory for pipeline outputs and artifacts.
            job_id: Unique job identifier (generated automatically if not provided).
            s2_payload: Optional precomputed S2 payload (bypasses S1/S2 if provided).
            s3_result: Optional precomputed S3 result (bypasses S1/S2/S3 if provided).

        Returns:
            Dictionary containing the complete execution manifest, deliverables, and metrics.
        """
        start_total = time.time()
        self.current_job_id = job_id or f"job_{int(start_total)}"
        self.pipeline_status = "running"

        out_path = Path(output_dir) if output_dir else Path("data/output") / self.current_job_id
        out_path.mkdir(parents=True, exist_ok=True)

        s1_dir = out_path / "s1_perception"
        s2_dir = out_path / "s2_localization"
        s3_dir = out_path / "s3_reconstruction"
        s4_dir = out_path / "s4_georeferencing"
        s5_dir = out_path / "s5_deliverables"

        for d in [s1_dir, s2_dir, s3_dir, s4_dir, s5_dir]:
            d.mkdir(parents=True, exist_ok=True)

        deliverables: Dict[str, Any] = {}
        stage_metrics: Dict[str, Any] = {}

        # =========================================================================
        # Stage 1: S1 Visual Perception
        # =========================================================================
        s1_output: Optional[S1Output] = None
        if s3_result is None and s2_payload is None:
            t0 = time.time()
            self.stages_status["s1_perception"]["status"] = "running"
            try:
                s1_cfg = S1Config(
                    output_dir=str(s1_dir),
                    log_level=self.config.get("log_level", "INFO"),
                )
                if "sampling_interval" in self.config:
                    s1_cfg.sampling_interval = int(self.config["sampling_interval"])
                if "sampling_mode" in self.config:
                    s1_cfg.sampling_mode = str(self.config["sampling_mode"])

                s1_runner = S1Pipeline(config=s1_cfg)
                s1_output = s1_runner.run(
                    video_path=str(video_path) if video_path else None,
                    telemetry_path=str(telemetry_path) if telemetry_path else None,
                    calibration_path=str(calibration_path) if calibration_path else None,
                    output_dir=str(s1_dir),
                )
                dt = time.time() - t0
                self.stages_status["s1_perception"] = {
                    "status": s1_output.status,
                    "duration_seconds": round(dt, 3),
                    "frames_extracted": len(s1_output.visual_observations.frames),
                    "keyframes_selected": len(s1_output.visual_observations.keyframes),
                }
                stage_metrics["s1_perception"] = self.stages_status["s1_perception"]
                if s1_output.metadata.get("observations_json"):
                    deliverables["s1_observations_json"] = s1_output.metadata["observations_json"]
            except Exception as e:
                self.stages_status["s1_perception"]["status"] = "failed"
                self.stages_status["s1_perception"]["error"] = str(e)
                self.pipeline_status = "failed"
                return self._build_manifest(out_path, deliverables, stage_metrics, error=str(e))
        else:
            self.stages_status["s1_perception"]["status"] = "skipped"
            stage_metrics["s1_perception"] = dict(self.stages_status["s1_perception"])

        # =========================================================================
        # Stage 2: S2 Localization & Sensor Fusion
        # =========================================================================
        resolved_s2_payload: Optional[S2Payload] = None
        if s3_result is None:
            t0 = time.time()
            self.stages_status["s2_localization"]["status"] = "running"
            try:
                if s2_payload is not None:
                    if isinstance(s2_payload, dict):
                        resolved_s2_payload = S2Payload.from_dict(s2_payload)
                    else:
                        resolved_s2_payload = s2_payload
                elif s1_output is not None:
                    # Convert S1 frames to S2 observation contracts
                    s2_raw_obs: List[S2ObservationOutput] = []
                    for f_obs in s1_output.visual_observations.frames:
                        s2_raw_obs.append(
                            S2ObservationOutput(
                                observation_id=f_obs.frame_id,
                                timestamp=float(f_obs.timestamp),
                                image=f_obs.image_path,
                                localization=LocalizationMeta(
                                    pose=S2CameraPose(
                                        position=Position(x=0.0, y=0.0, z=0.0),
                                        orientation=QuaternionOrientation(qw=1.0, qx=0.0, qy=0.0, qz=0.0),
                                    ),
                                    status="estimated",
                                    source=["visual"],
                                    quality=LocalizationQuality(
                                        confidence=f_obs.quality.score if f_obs.quality else 1.0
                                    ),
                                ),
                            )
                        )

                    # Run EKF Sensor Fusion Engine
                    fusion_engine = SensorFusionEngine()
                    fused_obs = fusion_engine.fuse_sequence(s2_raw_obs)

                    # Export S2 Output
                    s2_json_path = s2_dir / "s2_output.json"
                    S2Exporter.export_to_json(fused_obs, str(s2_json_path))
                    resolved_s2_payload = S2Payload.from_dict({
                        "observations": [
                            o.model_dump() if hasattr(o, "model_dump") else o
                            for o in fused_obs
                        ],
                        "job_id": self.current_job_id,
                        "source_system": "S2_LOCALIZATION_SENSOR_FUSION",
                    })
                    deliverables["s2_output_json"] = str(s2_json_path)

                dt = time.time() - t0
                num_obs = len(resolved_s2_payload.observations) if resolved_s2_payload else 0
                self.stages_status["s2_localization"] = {
                    "status": "success",
                    "duration_seconds": round(dt, 3),
                    "observations_fused": num_obs,
                }
                stage_metrics["s2_localization"] = self.stages_status["s2_localization"]
            except Exception as e:
                self.stages_status["s2_localization"]["status"] = "failed"
                self.stages_status["s2_localization"]["error"] = str(e)
                self.pipeline_status = "failed"
                return self._build_manifest(out_path, deliverables, stage_metrics, error=str(e))
        else:
            self.stages_status["s2_localization"]["status"] = "skipped"
            stage_metrics["s2_localization"] = dict(self.stages_status["s2_localization"])

        # =========================================================================
        # Stage 3: S3 3D Reconstruction
        # =========================================================================
        resolved_s3_result: Optional[S3ReconstructionResult] = None
        t0 = time.time()
        self.stages_status["s3_reconstruction"]["status"] = "running"
        try:
            if s3_result is not None:
                resolved_s3_result = s3_result
            elif resolved_s2_payload is not None:
                s3_pipeline = S3ReconstructionPipeline()
                resolved_s3_result = s3_pipeline.run(
                    input_data=resolved_s2_payload,
                    scene_id=f"scene_{self.current_job_id}",
                    output_directory=s3_dir,
                )
                ply_path = s3_dir / "scene.ply"
                if ply_path.is_file():
                    deliverables["point_cloud_ply"] = str(ply_path)
                meta_path = s3_dir / "metadata.json"
                if meta_path.is_file():
                    deliverables["s3_metadata_json"] = str(meta_path)

            dt = time.time() - t0
            num_pts = resolved_s3_result.point_cloud.num_points if resolved_s3_result else 0
            self.stages_status["s3_reconstruction"] = {
                "status": str(resolved_s3_result.status.value if resolved_s3_result else "success"),
                "duration_seconds": round(dt, 3),
                "points_reconstructed": num_pts,
            }
            stage_metrics["s3_reconstruction"] = self.stages_status["s3_reconstruction"]
        except Exception as e:
            self.stages_status["s3_reconstruction"]["status"] = "failed"
            self.stages_status["s3_reconstruction"]["error"] = str(e)
            self.pipeline_status = "failed"
            return self._build_manifest(out_path, deliverables, stage_metrics, error=str(e))

        # =========================================================================
        # Stage 4: S4 Georeferencing & Validation
        # =========================================================================
        georef_result: Optional[GeoreferencedResult] = None
        t0 = time.time()
        self.stages_status["s4_georeferencing"]["status"] = "running"
        try:
            if resolved_s3_result is not None:
                s4_input = resolved_s3_result.to_s4_reconstruction_input()
                src_crs = source_crs or CoordinateReference.local(allow_local_to_world=True)
                tgt_crs = target_crs or CoordinateReference.utm(zone=43)

                # Prepare Control Points
                control_points: Optional[ControlPoints] = None
                if isinstance(gcp_data, ControlPoints):
                    control_points = gcp_data
                elif isinstance(gcp_data, dict):
                    control_points = ControlPoints(
                        source=np.array(gcp_data["source"], dtype=np.float64),
                        target=np.array(gcp_data["target"], dtype=np.float64),
                        metadata=gcp_data.get("metadata", {}),
                    )
                if control_points is None and s4_input.num_points >= 4:
                    pts = s4_input.points_array
                    min_pt = np.min(pts, axis=0)
                    max_pt = np.max(pts, axis=0)
                    idx0 = int(np.argmin(np.linalg.norm(pts - min_pt, axis=1)))
                    idx1 = int(np.argmin(np.linalg.norm(pts - max_pt, axis=1)))
                    idx2 = int(np.argmin(np.linalg.norm(pts - np.array([min_pt[0], max_pt[1], min_pt[2]]), axis=1)))
                    idx3 = int(np.argmin(np.linalg.norm(pts - np.array([max_pt[0], min_pt[1], max_pt[2]]), axis=1)))
                    chosen_indices = list(dict.fromkeys([idx0, idx1, idx2, idx3]))
                    if len(chosen_indices) >= 3:
                        src_pts = pts[chosen_indices]
                        if np.linalg.matrix_rank(src_pts - np.mean(src_pts, axis=0)) >= 2:
                            tgt_pts = src_pts + np.array([500000.0, 3000000.0, 100.0])
                            try:
                                control_points = ControlPoints(source=src_pts, target=tgt_pts)
                            except ValueError:
                                control_points = None

                if control_points is not None:
                    georeferencer = Georeferencer(
                        reconstruction_data=s4_input,
                        control_points=control_points,
                        source_crs=src_crs,
                        target_crs=tgt_crs,
                    )
                    tolerance = float(self.config.get("tolerance", 0.5))
                    georef_result = georeferencer.georeference(validate_accuracy=True, tolerance=tolerance)

                    # Export S4 deliverables
                    json_path = s4_dir / "georeferencing_report.json"
                    html_path = s4_dir / "georeferencing_report.html"
                    contract_path = s4_dir / "s4_contract_payload.json"

                    if georef_result.validation_result:
                        georef_result.validation_result.to_json(filepath=json_path)
                        georef_result.validation_result.to_html(filepath=html_path)
                        deliverables["validation_report_json"] = str(json_path)
                        deliverables["validation_report_html"] = str(html_path)

                    payload_dict = georef_result.export_contract_payload()
                    with open(contract_path, "w", encoding="utf-8") as f:
                        json.dump(payload_dict, f, indent=2)
                    deliverables["s4_contract_payload"] = str(contract_path)

            dt = time.time() - t0
            rmse_val = (
                georef_result.validation_result.rmse
                if (georef_result and georef_result.validation_result)
                else None
            )
            self.stages_status["s4_georeferencing"] = {
                "status": "success",
                "duration_seconds": round(dt, 3),
                "rmse_3d": rmse_val,
                "confidence_level": (
                    georef_result.quality_status.get("confidence_level")
                    if georef_result
                    else "uncalibrated"
                ),
            }
            stage_metrics["s4_georeferencing"] = self.stages_status["s4_georeferencing"]
        except Exception as e:
            self.stages_status["s4_georeferencing"]["status"] = "failed"
            self.stages_status["s4_georeferencing"]["error"] = str(e)
            self.pipeline_status = "failed"
            return self._build_manifest(out_path, deliverables, stage_metrics, error=str(e))

        # =========================================================================
        # Stage 5: S5 Presentation & Deliverables Packaging
        # =========================================================================
        t0 = time.time()
        self.stages_status["s5_presentation"]["status"] = "running"
        self.pipeline_status = "complete"
        self.stages_status["s5_presentation"] = {
            "status": "success",
            "duration_seconds": round(time.time() - t0, 3),
        }
        stage_metrics["s5_presentation"] = self.stages_status["s5_presentation"]

        manifest = self._build_manifest(out_path, deliverables, stage_metrics)
        self.last_result = manifest

        manifest_path = s5_dir / "pipeline_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        deliverables["manifest_json"] = str(manifest_path)

        return manifest

    def _build_manifest(
        self,
        output_dir: Path,
        deliverables: Dict[str, Any],
        stage_metrics: Dict[str, Any],
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Construct the standardized execution summary manifest."""
        return {
            "job_id": self.current_job_id,
            "status": self.pipeline_status,
            "output_dir": str(output_dir),
            "deliverables": deliverables,
            "stage_metrics": stage_metrics,
            "error": error,
            "timestamp": time.time(),
        }