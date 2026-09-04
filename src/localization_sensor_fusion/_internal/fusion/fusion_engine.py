"""Extended Kalman Filter (EKF) sensor fusion for local UAV localization."""

from __future__ import annotations

import math
from typing import List, Optional, Protocol, Sequence, Tuple

import numpy as np

from ..schemas.contracts import (
    CameraPose,
    CameraPose,
    LocalizationQuality,
    Position,
    QuaternionOrientation,
    S2ObservationOutput,
)


class GeodeticToEnuTransformer(Protocol):
    """Minimal CRS interface needed by the fusion engine."""

    def geodetic_to_enu(self, lat: float, lon: float, alt: float) -> tuple[float, float, float]: ...


POSITION = slice(0, 3)
VELOCITY = slice(3, 6)
ACCELERATION = slice(6, 9)
QUATERNION = slice(9, 13)  # [qw, qx, qy, qz], body to ENU
ANGULAR_VELOCITY = slice(13, 16)
STATE_DIMENSION = 16
GRAVITY_ENU_MPS2 = np.array([0.0, 0.0, -9.80665], dtype=np.float64)


class SensorFusionEngine:
    """Fuse visual poses, IMU measurements, and GPS fixes in a local ENU frame."""

    def __init__(
        self,
        process_noise: float = 1e-3,
        measurement_noise: float = 1e-2,
        motion_model: str = "constant_acceleration",
        coordinate_transformer: GeodeticToEnuTransformer | None = None,
    ) -> None:
        if motion_model not in {"constant_velocity", "constant_acceleration"}:
            raise ValueError("motion_model must be 'constant_velocity' or 'constant_acceleration'")

        self.motion_model = motion_model
        self.coordinate_transformer = coordinate_transformer
        self.state = np.zeros((STATE_DIMENSION, 1), dtype=np.float64)
        self.state[QUATERNION, 0] = np.array([1.0, 0.0, 0.0, 0.0])
        self.covariance = np.eye(STATE_DIMENSION, dtype=np.float64)
        self.Q = np.eye(STATE_DIMENSION, dtype=np.float64) * process_noise
        self.default_R = np.eye(3, dtype=np.float64) * measurement_noise
        self.orientation_R = np.eye(4, dtype=np.float64) * measurement_noise
        self.F = np.eye(STATE_DIMENSION, dtype=np.float64)
        self.H = self._selection_matrix(POSITION)
        self.last_timestamp: Optional[float] = None

    # --- Covariance & Target 4 Quality Surfacing ---

    @property
    def position_covariance(self) -> np.ndarray:
        """Returns the 3x3 position covariance matrix (in ENU m²)."""
        return self.covariance[POSITION, POSITION].copy()

    @property
    def position_uncertainty_std(self) -> Tuple[float, float, float]:
        """Returns 1-sigma standard deviations for ENU position (sigma_x, sigma_y, sigma_z)."""
        pos_cov = self.position_covariance
        sigma_x = math.sqrt(max(0.0, float(pos_cov[0, 0])))
        sigma_y = math.sqrt(max(0.0, float(pos_cov[1, 1])))
        sigma_z = math.sqrt(max(0.0, float(pos_cov[2, 2])))
        return sigma_x, sigma_y, sigma_z

    def calculate_confidence_score(
        self,
        max_acceptable_std_m: float = 5.0,
    ) -> float:
        """Maps current EKF position standard deviation to a normalized confidence score in [0.0, 1.0]."""
        sx, sy, sz = self.position_uncertainty_std
        rms_std = math.sqrt((sx**2 + sy**2 + sz**2) / 3.0)

        confidence = math.exp(-1.0 * (rms_std / (max_acceptable_std_m / 2.0)))
        return float(np.clip(confidence, 0.0, 1.0))

    # --- Static Math Helpers ---

    @staticmethod
    def _as_vector(values: Sequence[float] | np.ndarray, size: int, name: str) -> np.ndarray:
        vector = np.asarray(values, dtype=np.float64).reshape(-1)
        if vector.size != size:
            raise ValueError(f"{name} must contain exactly {size} values")
        return vector

    @staticmethod
    def _normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(quaternion))
        if norm <= 1e-12:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return quaternion / norm

    @staticmethod
    def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        lw, lx, ly, lz = left
        rw, rx, ry, rz = right
        return np.array(
            [
                lw * rw - lx * rx - ly * ry - lz * rz,
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
            ],
            dtype=np.float64,
        )

    @classmethod
    def _integrate_quaternion(cls, quaternion: np.ndarray, angular_rate: np.ndarray, dt: float) -> np.ndarray:
        angle = float(np.linalg.norm(angular_rate) * dt)
        if angle <= 1e-12:
            return cls._normalize_quaternion(quaternion)
        axis = angular_rate / np.linalg.norm(angular_rate)
        delta = np.array([np.cos(angle / 2.0), *(axis * np.sin(angle / 2.0))], dtype=np.float64)
        return cls._normalize_quaternion(cls._quaternion_multiply(quaternion, delta))

    @staticmethod
    def _rotation_matrix_from_quaternion(quaternion: np.ndarray) -> np.ndarray:
        qw, qx, qy, qz = SensorFusionEngine._normalize_quaternion(quaternion)
        return np.array(
            [
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _selection_matrix(state_slice: slice) -> np.ndarray:
        width = state_slice.stop - state_slice.start
        matrix = np.zeros((width, STATE_DIMENSION), dtype=np.float64)
        matrix[:, state_slice] = np.eye(width, dtype=np.float64)
        return matrix

    # --- Internal Measurement Mechanics ---

    def _measurement_update(
        self,
        measurement: Sequence[float] | np.ndarray,
        predicted_measurement: Sequence[float] | np.ndarray,
        measurement_jacobian: np.ndarray,
        measurement_covariance: np.ndarray | None,
    ) -> None:
        z = np.asarray(measurement, dtype=np.float64).reshape(-1, 1)
        h = np.asarray(predicted_measurement, dtype=np.float64).reshape(-1, 1)
        H = np.asarray(measurement_jacobian, dtype=np.float64)
        if H.shape != (z.shape[0], STATE_DIMENSION):
            raise ValueError("measurement_jacobian has an invalid shape")
        R = self.default_R if measurement_covariance is None else np.asarray(measurement_covariance, dtype=np.float64)
        if R.shape != (z.shape[0], z.shape[0]):
            raise ValueError("measurement covariance has an invalid shape")

        innovation_covariance = H @ self.covariance @ H.T + R
        try:
            gain = np.linalg.solve(innovation_covariance, H @ self.covariance).T
        except np.linalg.LinAlgError as error:
            raise ValueError("measurement update has a singular innovation covariance") from error

        self.state += gain @ (z - h)
        self.state[QUATERNION, 0] = self._normalize_quaternion(self.state[QUATERNION, 0])
        residual = np.eye(STATE_DIMENSION) - gain @ H
        self.covariance = residual @ self.covariance @ residual.T + gain @ R @ gain.T
        self.covariance = 0.5 * (self.covariance + self.covariance.T)

    # --- Motion Model Prediction ---

    def predict(
        self,
        dt: float,
        gyro_rates: Sequence[float] | np.ndarray | None = None,
        acceleration_body_mps2: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        """Propagate CV/CA motion and quaternion attitude by one time step."""
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        position = self.state[POSITION, 0]
        velocity = self.state[VELOCITY, 0]
        acceleration = self.state[ACCELERATION, 0]
        quaternion = self.state[QUATERNION, 0]
        angular_rate = self.state[ANGULAR_VELOCITY, 0]
        if gyro_rates is not None:
            angular_rate = self._as_vector(gyro_rates, 3, "gyro_rates")
            self.state[ANGULAR_VELOCITY, 0] = angular_rate
        if acceleration_body_mps2 is not None:
            body_acceleration = self._as_vector(acceleration_body_mps2, 3, "acceleration_body_mps2")
            acceleration = self._rotation_matrix_from_quaternion(quaternion) @ body_acceleration + GRAVITY_ENU_MPS2
            self.state[ACCELERATION, 0] = acceleration

        self.F = np.eye(STATE_DIMENSION, dtype=np.float64)
        self.F[POSITION, VELOCITY] = np.eye(3) * dt
        if self.motion_model == "constant_acceleration":
            self.F[POSITION, ACCELERATION] = np.eye(3) * (0.5 * dt * dt)
            self.F[VELOCITY, ACCELERATION] = np.eye(3) * dt
            position = position + velocity * dt + 0.5 * acceleration * dt * dt
            velocity = velocity + acceleration * dt
        else:
            position = position + velocity * dt

        self.state[POSITION, 0] = position
        self.state[VELOCITY, 0] = velocity
        self.state[QUATERNION, 0] = self._integrate_quaternion(quaternion, angular_rate, dt)
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q * dt
        self.covariance = 0.5 * (self.covariance + self.covariance.T)

    # --- Sensor Updates ---

    def update(self, measurement: Sequence[float] | np.ndarray, R_custom: np.ndarray | None = None) -> None:
        """Backward-compatible local ENU position update."""
        self.update_gps_enu(measurement, R_custom)

    def update_gps_enu(
        self,
        position_enu_m: Sequence[float] | np.ndarray,
        covariance_enu_m2: np.ndarray | None = None,
    ) -> None:
        """Update the position state from a local ENU GPS fix in metres."""
        position = self._as_vector(position_enu_m, 3, "position_enu_m")
        covariance = self.default_R if covariance_enu_m2 is None else np.asarray(covariance_enu_m2, dtype=np.float64)
        self._measurement_update(position, self.state[POSITION, 0], self.H, covariance)

    def update_gps_geodetic(
        self,
        latitude_deg: float,
        longitude_deg: float,
        altitude_m: float,
        covariance_enu_m2: np.ndarray | None = None,
    ) -> None:
        """Convert a WGS84 fix through the configured CRS transformer and update ENU state."""
        if self.coordinate_transformer is None:
            raise ValueError("a coordinate_transformer is required for geodetic GPS updates")
        self.update_gps_enu(
            self.coordinate_transformer.geodetic_to_enu(latitude_deg, longitude_deg, altitude_m),
            covariance_enu_m2,
        )

    def update_accelerometer(
        self,
        acceleration_body_mps2: Sequence[float] | np.ndarray,
        covariance: np.ndarray | None = None,
    ) -> None:
        """Update ENU acceleration from a body-frame specific-force measurement."""
        measured = self._as_vector(acceleration_body_mps2, 3, "acceleration_body_mps2")
        rotation = self._rotation_matrix_from_quaternion(self.state[QUATERNION, 0])
        predicted = rotation.T @ (self.state[ACCELERATION, 0] - GRAVITY_ENU_MPS2)
        H = np.zeros((3, STATE_DIMENSION), dtype=np.float64)
        H[:, ACCELERATION] = rotation.T
        self._measurement_update(measured, predicted, H, covariance)

    def update_gyroscope(
        self,
        angular_velocity_body_radps: Sequence[float] | np.ndarray,
        covariance: np.ndarray | None = None,
    ) -> None:
        """Update body-frame angular velocity from a gyroscope measurement."""
        measured = self._as_vector(angular_velocity_body_radps, 3, "angular_velocity_body_radps")
        self._measurement_update(measured, self.state[ANGULAR_VELOCITY, 0], self._selection_matrix(ANGULAR_VELOCITY), covariance)

    def update_orientation(
        self,
        quaternion_scalar_first: Sequence[float] | np.ndarray,
        covariance: np.ndarray | None = None,
    ) -> None:
        """Fuse a visual scalar-first quaternion observation into the state."""
        measured = self._normalize_quaternion(self._as_vector(quaternion_scalar_first, 4, "quaternion_scalar_first"))
        R = self.orientation_R if covariance is None else covariance
        self._measurement_update(measured, self.state[QUATERNION, 0], self._selection_matrix(QUATERNION), R)

    def process_gps_fix(
        self,
        position: Sequence[float] | np.ndarray,
        covariance_matrix: np.ndarray | None = None,
    ) -> None:
        """Compatibility wrapper for callers that already supply ENU metres."""
        self.update_gps_enu(position, covariance_matrix)

    # --- Schema Observation Pipeline ---

    def get_observation_output(self, timestamp: float) -> S2ObservationOutput:
        """Generate structured S2 observation output incorporating state and covariance metrics."""
        pos = Position(
            x=float(self.state[POSITION, 0][0]),
            y=float(self.state[POSITION, 0][1]),
            z=float(self.state[POSITION, 0][2]),
        )

        qw, qx, qy, qz = self.state[QUATERNION, 0]
        orientation = QuaternionOrientation(qw=float(qw), qx=float(qx), qy=float(qy), qz=float(qz))

        quality = LocalizationQuality(
            confidence=self.calculate_confidence_score(),
            position_covariance=self.position_covariance.flatten().tolist(),
            is_valid=True,
        )

        return S2ObservationOutput.model_construct(
            observation_id=f"fused_{timestamp}",
            timestamp=timestamp,
            pose=CameraPose(position=pos, orientation=orientation),
            localization=quality,
        )

    def process_observation(
        self,
        observation: S2ObservationOutput | None,
        gyro_rates: Sequence[float] | np.ndarray | None = None,
        acceleration_body_mps2: Sequence[float] | np.ndarray | None = None,
        gps_enu_m: Sequence[float] | np.ndarray | None = None,
        gps_std_dev: Sequence[float] | np.ndarray | None = None,
    ) -> S2ObservationOutput | None:
        """Fuse one observation and return it with position, attitude, and quality metrics from the EKF."""
        if observation is None:
            return None
        timestamp = float(observation.timestamp)
        dt = max(timestamp - self.last_timestamp, 1e-3) if self.last_timestamp is not None else 0.1
        self.last_timestamp = timestamp
        self.predict(dt, gyro_rates=gyro_rates, acceleration_body_mps2=acceleration_body_mps2)
        if gyro_rates is not None:
            self.update_gyroscope(gyro_rates)
        if acceleration_body_mps2 is not None:
            self.update_accelerometer(acceleration_body_mps2)

        covariance = None
        if gps_std_dev is not None:
            standard_deviation = self._as_vector(gps_std_dev, 3, "gps_std_dev")
            covariance = np.diag(standard_deviation**2)
        if gps_enu_m is not None:
            self.update_gps_enu(gps_enu_m, covariance)
        elif observation.pose and observation.pose.position:
            position = observation.pose.position
            self.update_gps_enu([position.x, position.y, position.z], covariance)
        if observation.pose and observation.pose.orientation:
            orientation = observation.pose.orientation
            self.update_orientation([orientation.qw, orientation.qx, orientation.qy, orientation.qz])

        qw, qx, qy, qz = self.state[QUATERNION, 0]
        observation.pose = CameraPose(
            position=Position(
                x=float(self.state[POSITION, 0][0]),
                y=float(self.state[POSITION, 0][1]),
                z=float(self.state[POSITION, 0][2]),
            ),
            orientation=QuaternionOrientation(qw=float(qw), qx=float(qx), qy=float(qy), qz=float(qz)),
        )

        observation.localization = LocalizationQuality(
            confidence=self.calculate_confidence_score(),
            position_covariance=self.position_covariance.flatten().tolist(),
            is_valid=True,
        )
        return observation
    def fuse_sequence(self, observations: List[S2ObservationOutput]) -> List[S2ObservationOutput]:
        return [fused for observation in observations if (fused := self.process_observation(observation)) is not None]