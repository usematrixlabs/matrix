"""Localization and Sensor Fusion Subsystem Package.

Public entry points used by the orchestrator and downstream adapters:

- ``S1InputAdapter``        — validates S1 observations into the S2 input contract
- ``S2InputAdapter``        — adapts raw localized frames into ``S2ObservationOutput``
- ``VisualLocalizerEngine`` — single-frame PnP pose estimation + ORB extraction
- ``TrajectorySmoother``    — moving-average trajectory filter
- ``SensorFusionEngine``    — EKF-based visual+IMU+GPS fusion
- ``S2Exporter``            — serializes the canonical ``S2PayloadOutput`` to JSON
- ``build_s2_payload``      — S2 → S3 bridge: builds an S3-ready S2Payload with
                              matched 2D feature tracks derived from the
                              localized S2 observations
- Contracts                 — Pydantic models shared across S1/S2/S3 boundaries
"""

from .adapters.s1_adapter import S1AdapterValidationError, S1InputAdapter
from .adapters.s2_bridge import build_s2_payload, build_s2_payload_from_s2
from .adapters.s2_input_adapter import S2InputAdapter
from .engines.colmap_engine import VisualLocalizerEngine, ColmapLocalizationEngine
from .engines.trajectory_smoother import TrajectorySmoother
from .fusion.fusion_engine import SensorFusionEngine
from .exporters.s2_exporter import S2Exporter
from .schemas.contracts import (
    CameraInfo,
    CameraIntrinsics,
    CameraPose,
    Distortion,
    FusedState,
    FrameQuality,
    LocalizationMeta,
    LocalizationQuality,
    LocalizationSource,
    PoseStatus,
    Position,
    QualityStatus,
    QuaternionOrientation,
    S1ObservationInput,
    S2ObservationOutput,
    S2PayloadOutput,
    Units,
)

__all__ = [
    # Adapters
    "S1InputAdapter",
    "S1AdapterValidationError",
    "S2InputAdapter",
    "build_s2_payload",
    "build_s2_payload_from_s2",
    # Engines
    "VisualLocalizerEngine",
    "ColmapLocalizationEngine",
    "TrajectorySmoother",
    "SensorFusionEngine",
    # Exporter
    "S2Exporter",
    # Contracts
    "CameraInfo",
    "CameraIntrinsics",
    "CameraPose",
    "Distortion",
    "FusedState",
    "FrameQuality",
    "LocalizationMeta",
    "LocalizationQuality",
    "LocalizationSource",
    "PoseStatus",
    "Position",
    "QualityStatus",
    "QuaternionOrientation",
    "S1ObservationInput",
    "S2ObservationOutput",
    "S2PayloadOutput",
    "Units",
]
