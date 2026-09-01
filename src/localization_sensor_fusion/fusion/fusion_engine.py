"""Extended Kalman Filter (EKF) Sensor Fusion Engine for UAV Localization."""

import numpy as np
from typing import List, Optional
from src.localization_sensor_fusion.schemas.contracts import (
    S2ObservationOutput,
    Position,
    QuaternionOrientation,
    CameraPose,
    LocalizationMeta,
)


class SensorFusionEngine:
    """Fuses visual localization data with telemetry/IMU inputs using an EKF."""

    def __init__(self, process_noise: float = 1e-3, measurement_noise: float = 1e-2):
        # 6-DOF State vector: [x, y, z, vx, vy, vz]^T as (6, 1) column matrix
        self.state = np.zeros((6, 1), dtype=np.float64)

        # State Covariance matrix (P)
        self.covariance = np.eye(6, dtype=np.float64) * 1.0

        # Process noise covariance (Q)
        self.Q = np.eye(6, dtype=np.float64) * process_noise

        # Default Measurement noise covariance (R) for position [x, y, z]
        self.default_R = np.eye(3, dtype=np.float64) * measurement_noise

        # State Transition matrix (F)
        self.F = np.eye(6, dtype=np.float64)

        # Measurement matrix (H) - maps 6D state to 3D position [x, y, z]
        self.H = np.zeros((3, 6), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        self.last_timestamp: Optional[float] = None

    def predict(self, dt: float) -> None:
        """Prediction step using constant velocity motion model."""
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        # State prediction: x = F * x
        self.state = self.F @ self.state

        # Covariance prediction: P = F * P * F^T + Q
        self.covariance = (self.F @ self.covariance @ self.F.T) + self.Q

    def update(
        self, 
        measurement: np.ndarray, 
        R_custom: Optional[np.ndarray] = None
    ) -> None:
        """
        Update step using visual position measurements and dynamic measurement covariance.

        :param measurement: 3D position array [x, y, z] or (3, 1) matrix
        :param R_custom: Optional 3x3 custom measurement noise matrix (e.g., derived from GPS HDOP)
        """
        z = np.ascontiguousarray(measurement, dtype=np.float64).reshape(3, 1)
        R = R_custom if R_custom is not None else self.default_R

        # Residual / Innovation: y = z - H * x
        y = z - (self.H @ self.state)

        # Innovation covariance: S = H * P * H^T + R
        S = (self.H @ self.covariance @ self.H.T) + R

        try:
            K = self.covariance @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return

        # Update state estimate
        self.state = self.state + (K @ y)

        # Update covariance matrix P = (I - K*H) * P
        I = np.eye(6, dtype=np.float64)
        self.covariance = (I - (K @ self.H)) @ self.covariance

        # Enforce covariance matrix symmetry
        self.covariance = 0.5 * (self.covariance + self.covariance.T)

    def process_observation(
        self, 
        observation: S2ObservationOutput,
        gyro_rates: Optional[tuple] = None,
        gps_std_dev: Optional[tuple] = None
    ) -> S2ObservationOutput:
        """
        Processes a single observation through the EKF pipeline, updating state and pose.
        
        Returns the observation with updated pose from the EKF state.
        """
        current_time = observation.timestamp

        if self.last_timestamp is not None:
            dt = max(current_time - self.last_timestamp, 1e-3)
        else:
            dt = 0.1  # Default time step for the first frame

        self.last_timestamp = current_time

        # 1. Kinematic State Prediction
        self.predict(dt)

        # 2. Measurement Update if pose available
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
                # Scale measurement noise inversely with visual confidence score
                scaled_var = self.default_R[0, 0] / max(observation.localization.quality.confidence, 0.01)
                R_dynamic = np.eye(3, dtype=np.float64) * scaled_var

            # 3. Measurement Update
            self.update(pos_meas, R_custom=R_dynamic)

            # 4. Mutate observation pose with updated EKF position
            observation.pose = CameraPose(
                position=Position(
                    x=float(self.state[0, 0]),
                    y=float(self.state[1, 0]),
                    z=float(self.state[2, 0]),
                ),
                orientation=observation.pose.orientation if observation.pose.orientation else QuaternionOrientation(
                    qx=0.0, qy=0.0, qz=0.0, qw=1.0
                ),
            )

        return observation

    def fuse_sequence(
        self, 
        observations: List[S2ObservationOutput]
    ) -> List[S2ObservationOutput]:
        """Runs the EKF sequentially across an entire list of observation outputs."""
        return [self.process_observation(obs) for obs in observations]