from __future__ import annotations

from enum import Enum
import numpy as np
from pydantic import BaseModel, Field


class QualityStatus(str, Enum):
    GOOD = "GOOD"
    BLURRY = "BLURRY"
    OVEREXPOSED = "OVEREXPOSED"
    UNDEREXPOSED = "UNDEREXPOSED"
    LOW_FEATURE = "LOW_FEATURE"
    CORRUPTED = "CORRUPTED"


class PoseStatus(str, Enum):
    ESTIMATED = "estimated"
    UNCERTAIN = "uncertain"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"


class LocalizationSource(str, Enum):
    VISUAL = "visual"
    GPS = "gps"
    VISUAL_GPS = "visual_gps"
    VISUAL_IMU = "visual_imu"


class CameraIntrinsics(BaseModel):
    fx: float
    fy: float
    cx: float
    cy: float


class Distortion(BaseModel):
    model: str = "opencv"
    coefficients: list[float] = Field(default_factory=list)


class CameraInfo(BaseModel):
    width: int | None = None
    height: int | None = None
    intrinsics: CameraIntrinsics | None = None
    distortion: Distortion | None = None


class FrameQuality(BaseModel):
    status: QualityStatus = QualityStatus.GOOD
    blur_score: float | None = None


class S1ObservationInput(BaseModel):
    observation_id: str
    timestamp: float
    image: str
    camera: CameraInfo | None = None
    quality: FrameQuality | None = None


class Position(BaseModel):
    x: float
    y: float
    z: float


class QuaternionOrientation(BaseModel):
    qx: float
    qy: float
    qz: float
    qw: float

    def to_numpy_scalar_first(self) -> np.ndarray:
        """Returns array in scalar-first format: [qw, qx, qy, qz] (Eigen / PyTorch)."""
        return np.array([self.qw, self.qx, self.qy, self.qz], dtype=np.float64)

    def to_numpy_scalar_last(self) -> np.ndarray:
        """Returns array in scalar-last format: [qx, qy, qz, qw] (SciPy / ROS)."""
        return np.array([self.qx, self.qy, self.qz, self.qw], dtype=np.float64)


class FusedState(BaseModel):
    position: Position
    orientation: QuaternionOrientation
    velocity: Position


class CameraPose(BaseModel):
    position: Position
    orientation: QuaternionOrientation


class LocalizationQuality(BaseModel):
    confidence: float = Field(..., ge=0.0, le=1.0)


class LocalizationMeta(BaseModel):
    status: PoseStatus
    source: list[LocalizationSource]
    quality: LocalizationQuality


class S2ObservationOutput(BaseModel):
    observation_id: str
    timestamp: float
    image: str
    camera: CameraInfo | None = None
    pose: CameraPose | None = None
    localization: LocalizationMeta


class Units(BaseModel):
    position: str = "m"
    rotation: str = "quaternion"


class S2PayloadOutput(BaseModel):
    observations: list[S2ObservationOutput]
    schema_version: str = "0.1"
    coordinate_frame: str = "local"
    units: Units = Field(default_factory=Units)