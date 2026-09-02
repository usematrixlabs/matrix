"""Extended Kalman Filter (EKF) Sensor Fusion Engine for UAV Localization."""

import numpy as np
from typing import List, Optional
from src.localization_sensor_fusion.schemas.contracts import (
    S2ObservationOutput,
    Position,
    QuaternionOrientation,
    CameraPose,
    LocalizationQuality,
)


class SensorFusionEngine:
    """Fuses visual localization data with telemetry/IMU inputs using an EKF."""

    def __init__(self, process_noise: float = 1e-3, measurement_noise: float = 1e-2):
        # 6-DOF Kinematic State vector: [x, y, z, vx, vy, vz]^T as (6, 1) column matrix
        self.state = np.zeros((6, 1), dtype=np.float64)
        self.covariance = np.eye(6, dtype=np.float64) * 1.0
        self.Q = np.eye(6, dtype=np.float64) * process_noise
        self.default_R = np.eye(3, dtype=np.float64) * measurement_noise

        self.F = np.eye(6, dtype=np.float64)
        self.H = np.zeros((3, 6), dtype=np.float64)
        self.H[0, 0], self.H[1, 1], self.H[2, 2] = 1.0, 1.0, 1.0

        self.last_timestamp: Optional[float] = None
        
        # Internal orientation tracking quaternion [qw, qx, qy, qz]
        self.current_orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def predict(self, dt: float, gyro_rates: Optional[tuple] = None) -> None:
        """
        Prediction step using constant velocity motion model and optional IMU gyro propagation.
        
        :param dt: Time step delta
        :param gyro_rates: Optional tuple of angular velocities (wx, wy, wz) in rad/s
        """
        # 1. Linear Kinematic Prediction (Position & Velocity)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        self.state = self.F @ self.state
        self.covariance = (self.F @ self.covariance @ self.F.T) + self.Q

        # 2. IMU Gyro Orientation Propagation (if angular rates provided)
        if gyro_rates is not None:
            wx, wy, wz = gyro_rates
            omega_mat = 0.5 * np.array([
                [0.0, -wx, -wy, -wz],
                [wx,  0.0,  wz, -wy],
                [wy, -wz,  0.0,  wx],
                [wz,  wy, -wx,  0.0]
            ], dtype=np.float64)
            
            q_vec = self.current_orientation
            q_dot = omega_mat @ q_vec
            q_new = q_vec + q_dot * dt
            
            # Robust quaternion normalization with zero-collapse safety
            norm = np.linalg.norm(q_new)
            if norm > 1e-8:
                self.current_orientation = q_new / norm

    def update(
        self, 
        measurement: np.ndarray, 
        R_custom: Optional[np.ndarray] = None
    ) -> None:
        """
        Update step using position measurements and dynamic measurement covariance.

        :param measurement: 3D position array [x, y, z] or (3, 1) matrix
        :param R_custom: Optional 3x3 custom measurement noise matrix
        """
        z = np.ascontiguousarray(measurement, dtype=np.float64).reshape(3, 1)
        R = R_custom if R_custom is not None else self.default_R

        y = z - (self.H @ self.state)
        S = (self.H @ self.covariance @ self.H.T) + R

        try:
            K = self.covariance @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return

        self.state = self.state + (K @ y)
        I = np.eye(6, dtype=np.float64)
        self.covariance = (I - (K @ self.H)) @ self.covariance

        # Enforce strict covariance matrix symmetry: P = 0.5 * (P + P^T)
        self.covariance = 0.5 * (self.covariance + self.covariance.T)

    def process_gps_fix(
        self, 
        position: np.ndarray, 
        covariance_matrix: Optional[np.ndarray] = None
    ) -> None:
        """
        Processes a raw external GPS fix with explicit per-fix covariance weighting.
        
        :param position: 3D position array [x, y, z] from GPS receiver
        :param covariance_matrix: Optional 3x3 covariance matrix (e.g., derived from HDOP/VDOP)
        """
        pos_meas = np.ascontiguousarray(position, dtype=np.float64).flatten()
        if len(pos_meas) != 3:
            raise ValueError("GPS position measurement must contain 3 elements [x, y, z]")

        R = covariance_matrix if covariance_matrix is not None else self.default_R
        R = np.ascontiguousarray(R, dtype=np.float64)
        if R.shape != (3, 3):
            R = np.eye(3, dtype=np.float64) * float(R[0, 0])

        self.update(pos_meas, R_custom=R)

    def process_observation(
        self, 
        observation: S2ObservationOutput,
        gyro_rates: Optional[tuple] = None,
        gps_std_dev: Optional[tuple] = None
    ) -> S2ObservationOutput:
        # Graceful handling of null/missing observations (Item 10)
        if observation is None:
            return None
        """
        Processes a single observation through the EKF pipeline, updating state and pose.
        """
        current_time = observation.timestamp

        if self.last_timestamp is not None:
            dt = max(current_time - self.last_timestamp, 1e-3)
        else:
            dt = 0.1  # Default time step for the first frame

        current_time = observation.timestamp if observation.timestamp else 0.0
        dt = max(current_time - self.last_timestamp, 1e-3) if self.last_timestamp is not None else 0.1
        self.last_timestamp = current_time

        # 1. Kinematic State Prediction
        self.predict(dt, gyro_rates=gyro_rates)

        # Update orientation tracking from visual observation if available
        if observation.pose and observation.pose.orientation:
            q = observation.pose.orientation
            q_arr = np.array([q.qw, q.qx, q.qy, q.qz], dtype=np.float64)
            q_norm = np.linalg.norm(q_arr)
            if q_norm > 1e-8:
                self.current_orientation = q_arr / q_norm

        # 2. Measurement Update if position available
        if observation.pose and observation.pose.position:
            pos_meas = np.array([
                observation.pose.position.x, 
                observation.pose.position.y, 
                observation.pose.position.z
            ], dtype=np.float64)

            R_dynamic = None
            if gps_std_dev is not None:
                sx, sy, sz = gps_std_dev
                R_dynamic = np.diag([sx**2, sy**2, sz**2]).astype(np.float64)
            elif (
                observation.localization 
                and observation.localization.quality 
                and observation.localization.quality.confidence > 0
            ):
                scaled_var = self.default_R[0, 0] / max(observation.localization.quality.confidence, 0.01)
                R_dynamic = np.eye(3, dtype=np.float64) * scaled_var

            self.update(pos_meas, R_custom=R_dynamic)

        # 3. Construct output observation pose from updated EKF state and orientation
        qw, qx, qy, qz = self.current_orientation
        observation.pose = CameraPose(
            position=Position(
                x=float(self.state[0, 0]),
                y=float(self.state[1, 0]),
                z=float(self.state[2, 0]),
            ),
            orientation=QuaternionOrientation(qw=qw, qx=qx, qy=qy, qz=qz),
        )

        return observation

    def fuse_sequence(self, observations: List[S2ObservationOutput]) -> List[S2ObservationOutput]:
        return [self.process_observation(obs) for obs in observations if obs is not None]