"""
S3 Reconstruction Pipeline

Coordinates end-to-end processing from S2 input ingestion, validation,
and data preparation through multi-view triangulation, quality assessment,
and artifact packaging.
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
    ) -> None:
        """
        Initialize the S3 pipeline.

        Parameters:
            engine: Optional custom reconstruction engine instance.
            max_reprojection_error_px: Threshold for reprojection error filtering.
            filter_statistical_outliers: If True, applies statistical outlier removal.
            check_image_files: If True, verifies image files on disk during input validation.
        """
        self.loader = S2InputLoader()
        self.validator = S2InputValidator(check_image_files=check_image_files)
        self.preparer = ReconstructionDataPreparer()
        self.engine = engine if engine is not None else DefaultReconstructionEngine(
            max_reprojection_error_px=max_reprojection_error_px
        )
        self.quality_evaluator = QualityEvaluator(
            max_acceptable_mean_reproj_px=max_reprojection_error_px
        )
        self.filter_statistical_outliers = filter_statistical_outliers

    def run(
        self,
        input_data: Union[str, Path, Dict[str, Any], S2Payload],
        scene_id: str = "scene_001",
        output_directory: Optional[Union[str, Path]] = None,
        raise_on_invalid_input: bool = False,
    ) -> S3ReconstructionResult:
        """
        Execute the full S3 reconstruction pipeline.

        Parameters:
            input_data: JSON file path, payload dictionary, or S2Payload instance.
            scene_id: Identifier for the output reconstructed scene.
            output_directory: Optional destination folder to save scene.ply and metadata.json.
            raise_on_invalid_input: If True, raises ValueError upon validation failure.

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
                metadata={"validation_errors": val_report.errors, "validation_warnings": val_report.warnings},
            )

        # 3. Input Preparation
        prepared = self.preparer.prepare(payload)

        # 4. Reconstruction Engine Execution
        points_3d, colors, reproj_errors, engine_stats = self.engine.reconstruct(prepared)

        # 5. Point Cloud Construction & Filtering
        point_cloud = PointCloudData(
            points=points_3d,
            colors=colors,
        )

        if self.filter_statistical_outliers and point_cloud.num_points > 15:
            point_cloud = PointCloudProcessor.statistical_outlier_removal(point_cloud)

        # 6. Quality Assessment & Status Classification
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

        # 7. Package Result
        spatial_ref = SpatialReference(
            coordinate_frame="S3_LOCAL",
            units=payload.units or "meters",
        )

        result = S3ReconstructionResult(
            scene_id=scene_id,
            job_id=job_id,
            status=status,
            point_cloud=point_cloud,
            spatial_reference=spatial_ref,
            quality=quality,
            failure_info=failure_info,
            metadata={
                "validation_warnings": val_report.warnings,
                "engine_stats": engine_stats,
            },
        )

        # 8. Export to Disk if requested
        if output_directory is not None:
            S3OutputPackager.package_to_directory(result, output_directory)

        return result

