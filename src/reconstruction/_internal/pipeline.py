"""
S3 Reconstruction Pipeline

Coordinates end-to-end processing from S2 input ingestion, validation,
and data preparation through multi-view triangulation, quality assessment,
and artifact packaging.

Calibration
-----------
The pipeline accepts an optional :class:`CameraCalibration`. When
provided, the preparer undistorts observation pixel coordinates before
triangulation using :func:`cv2.undistortPoints`, and the resulting
output ``metadata.json`` records the calibration provenance and
whether undistortion was applied.

The pipeline does not silently re-interpret the calibration: if the
calibration's resolution does not match the observed video resolution
the caller is responsible for explicitly scaling via
:meth:`CameraCalibration.scale_to_resolution` (the matrix dimensions
of every observation's underlying image determine the match). The
pipeline performs a final consistency check and raises a clear error
if a mismatch is detected.
"""

from pathlib import Path
import time
from typing import Any, Dict, Optional, Union
import numpy as np

from .engine.base import ReconstructionEngineBase
from .engine.reconstruct import DefaultReconstructionEngine
from .geometry.pointcloud import PointCloudProcessor
from .input.loader import S2InputLoader
from .input.validator import S2InputValidator
from .models.calibration import CameraCalibration, CameraCalibrationError
from .models.s3_output import PointCloudData, ReconstructionQuality, S3ReconstructionResult, SpatialReference
from .models.schema import S2Payload, S3Status
from .output.packaging import S3OutputPackager
from .preprocessing.prepare import ReconstructionDataPreparer
from .quality.evaluator import QualityEvaluator


class S3ReconstructionPipeline:
    """
    End-to-end pipeline orchestrator for Subsystem S3 (3D Reconstruction).
    """

    def __init__(
        self,
        engine: Optional[ReconstructionEngineBase] = None,
        max_reprojection_error_px: float = 3.0,
        filter_statistical_outliers: bool = True,
        check_image_files: bool = False,
        calibration: Optional[CameraCalibration] = None,
    ) -> None:
        """
        Initialize the S3 pipeline.

        Parameters:
            engine: Optional custom reconstruction engine instance.
            max_reprojection_error_px: Threshold for reprojection error filtering.
            filter_statistical_outliers: If True, applies statistical outlier removal.
            check_image_files: If True, verifies image files on disk during input validation.
            calibration: Optional camera calibration applied to undistort
                observations before triangulation. See
                :class:`CameraCalibration`.
        """
        self.loader = S2InputLoader()
        self.validator = S2InputValidator(check_image_files=check_image_files)
        self.calibration = calibration
        self.preparer = ReconstructionDataPreparer(calibration=self.calibration)
        self.engine = engine if engine is not None else DefaultReconstructionEngine(
            max_reprojection_error_px=max_reprojection_error_px
        )
        self.quality_evaluator = QualityEvaluator(
            max_acceptable_mean_reproj_px=max_reprojection_error_px
        )
        self.filter_statistical_outliers = filter_statistical_outliers

    def set_calibration(self, calibration: Optional[CameraCalibration]) -> None:
        """Attach (or clear) the calibration used for undistortion."""
        self.calibration = calibration
        self.preparer.set_calibration(calibration)

    def run(
        self,
        input_data: Union[str, Path, Dict[str, Any], S2Payload],
        scene_id: str = "scene_001",
        output_directory: Optional[Union[str, Path]] = None,
        raise_on_invalid_input: bool = False,
        video_resolution: Optional[tuple] = None,
    ) -> S3ReconstructionResult:
        """
        Execute the full S3 reconstruction pipeline.

        Parameters:
            input_data: JSON file path, payload dictionary, or S2Payload instance.
            scene_id: Identifier for the output reconstructed scene.
            output_directory: Optional destination folder to save scene.ply and metadata.json.
            raise_on_invalid_input: If True, raises ValueError upon validation failure.
            video_resolution: Optional ``(width, height)`` of the actual video
                frames. Used to verify that the supplied calibration matches
                the video dimensions (or has been pre-scaled). When omitted
                the pipeline infers it from the observation images; if it
                cannot be inferred and a calibration is set, the calibration
                is still used as-is and a warning is recorded.

        Returns:
            S3ReconstructionResult containing point cloud, quality, and metadata.
        """
        start_time = time.perf_counter()

        # 1. Ingestion
        if isinstance(input_data, S2Payload):
            payload = input_data
        elif isinstance(input_data, (str, Path)):
            payload = self.loader.load_from_file(input_data)
        elif isinstance(input_data, dict):
            payload = self.loader.load_from_dict(input_data)
        else:
            raise TypeError(f"Unsupported input_data type: {type(input_data).__name__}")

        job_id = payload.job_id or "job_unspecified"

        # 2. Boundary Validation
        val_report = self.validator.validate(payload)
        if not val_report.is_valid:
            if raise_on_invalid_input:
                val_report.raise_if_invalid()

            elapsed = time.perf_counter() - start_time
            empty_cloud = PointCloudData(points=np.empty((0, 3), dtype=np.float64))
            empty_quality = ReconstructionQuality(
                input_observations_count=len(payload.observations),
                processing_time_seconds=elapsed,
            )
            return S3ReconstructionResult(
                scene_id=scene_id,
                job_id=job_id,
                status=S3Status.INVALID_INPUT,
                point_cloud=empty_cloud,
                quality=empty_quality,
                failure_info="; ".join(val_report.errors),
                metadata={
                    "validation_errors": val_report.errors,
                    "validation_warnings": val_report.warnings,
                    "camera_calibration": self.calibration.to_dict() if self.calibration else None,
                },
            )

        # 3. Calibration / video-resolution consistency
        calib_resolution_warnings: list[str] = []
        if self.calibration is not None and video_resolution is not None:
            vw, vh = int(video_resolution[0]), int(video_resolution[1])
            if (vw, vh) != (self.calibration.image_width, self.calibration.image_height):
                raise CameraCalibrationError(
                    f"Calibration resolution {self.calibration.image_width}x"
                    f"{self.calibration.image_height} does not match video "
                    f"resolution {vw}x{vh}. The caller must explicitly scale "
                    f"the calibration via CameraCalibration.scale_to_resolution "
                    f"or supply a calibration matching the video."
                )

        # 4. Input Preparation (with optional undistortion)
        prepared = self.preparer.prepare(payload)

        # 5. Reconstruction Engine Execution
        points_3d, colors, reproj_errors, engine_stats = self.engine.reconstruct(prepared)

        # 6. Point Cloud Construction & Filtering
        point_cloud = PointCloudData(
            points=points_3d,
            colors=colors,
        )

        if self.filter_statistical_outliers and point_cloud.num_points > 15:
            point_cloud = PointCloudProcessor.statistical_outlier_removal(point_cloud)

        # 7. Quality Assessment & Status Classification
        elapsed = time.perf_counter() - start_time
        quality, status, failure_info = self.quality_evaluator.evaluate(
            points=point_cloud.points,
            reprojection_errors=reproj_errors,
            total_observations=prepared.total_observations,
            processed_observations=prepared.usable_observations,
            total_tracks=len(prepared.tracks),
            processing_time_s=elapsed,
            pre_validation_status=val_report.status,
        )

        # 8. Package Result
        spatial_ref = SpatialReference(
            coordinate_frame="S3_LOCAL",
            units=payload.units or "meters",
        )

        # Compose per-stage metadata including calibration provenance.
        result_metadata: Dict[str, Any] = {
            "validation_warnings": list(val_report.warnings) + calib_resolution_warnings,
            "engine_stats": engine_stats,
        }
        if self.calibration is not None:
            undistortion_applied = bool(prepared.metadata.get("undistortion_applied", False))
            result_metadata["camera_calibration"] = {
                **self.calibration.to_dict(),
                "undistortion_applied": undistortion_applied,
                "source": self.calibration.source,
            }
            # High-level summary block for quick S4/S5 inspection.
            result_metadata["camera_calibration_summary"] = {
                "camera_name": self.calibration.camera_name,
                "image_width": int(self.calibration.image_width),
                "image_height": int(self.calibration.image_height),
                "distortion_model": self.calibration.distortion_model,
                "distortion_applied": undistortion_applied,
            }

        result = S3ReconstructionResult(
            scene_id=scene_id,
            job_id=job_id,
            status=status,
            point_cloud=point_cloud,
            spatial_reference=spatial_ref,
            quality=quality,
            failure_info=failure_info,
            metadata=result_metadata,
        )

        # 9. Export to Disk if requested
        if output_directory is not None:
            S3OutputPackager.package_to_directory(result, output_directory)

        return result
