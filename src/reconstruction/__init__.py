"""
S3 — 3D Reconstruction Subsystem

Transforms visual observations and camera localization into 3D representations.
"""

from .engine import DefaultReconstructionEngine, MultiViewTriangulator, ReconstructionEngineBase
from .geometry import PlyIO, PointCloudProcessor
from .input import S2InputLoader, S2InputValidator, ValidationReport
from .models import (
    BoundingBox3D,
    CameraIntrinsics,
    CameraPose,
    FeatureObservation,
    LocalizationInfo,
    PointCloudData,
    ReconstructionQuality,
    S2Observation,
    S2Payload,
    S3ReconstructionResult,
    S3Status,
    SpatialReference,
)
from .pipeline import S3ReconstructionPipeline
from .preprocessing import PreparedReconstructionData, PreparedTrack, ReconstructionDataPreparer
from .quality import QualityEvaluator, S3FailureReason, S3ReconstructionError
from .reconstructor import Reconstructor

__all__ = [
    "Reconstructor",
    "S3ReconstructionPipeline",
    "DefaultReconstructionEngine",
    "MultiViewTriangulator",
    "ReconstructionEngineBase",
    "S2InputLoader",
    "S2InputValidator",
    "ValidationReport",
    "ReconstructionDataPreparer",
    "PreparedTrack",
    "PreparedReconstructionData",
    "PlyIO",
    "PointCloudProcessor",
    "QualityEvaluator",
    "S3FailureReason",
    "S3ReconstructionError",
    "S3Status",
    "CameraIntrinsics",
    "CameraPose",
    "LocalizationInfo",
    "FeatureObservation",
    "S2Observation",
    "S2Payload",
    "BoundingBox3D",
    "SpatialReference",
    "ReconstructionQuality",
    "PointCloudData",
    "S3ReconstructionResult",
]