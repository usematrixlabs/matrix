"""Extended Kalman Filter (EKF) Sensor Fusion Engine for UAV Localization."""

import numpy as np
from typing import List, Dict, Any, Optional
from src.localization_sensor_fusion.schemas.contracts import (
    S2ObservationOutput,
    CameraPose,
    Position,
    QuaternionOrientation,
    LocalizationQuality,
    LocalizationMeta,
)


class SensorFusionEngine:
    """Fuses visual localization data (e.g., COLMAP) with telemetry/IMU inputs using an EKF."""

    def __init__(self, process_noise: float = 1e-3, measurement_noise: float = 1e-2):
        # State vector: [x, y, z, vx, vy, vz]
        self.state = np.zeros((6, 1))
        
        # Covariance matrix
        self.covariance = np.eye(6) * 1.0
        
        # Process noise covariance (Q)
        self.Q = np.eye(6) * process_noise
        
        # Measurement noise covariance (R) for position [x, y, z]
        self.R = np.eye(3) * measurement_noise
        
        # Transition matrix (constant velocity model placeholder)
        self.F = np.eye(6)
        
        # Measurement matrix (mapping state to position measurement)
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        self.last_timestamp: Optional[float] = None

    def predict(self, dt: float) -> None:
        """Prediction step using constant velocity motion model."""
        # Update transition matrix with dt for velocity terms
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        # State prediction: x = F * x
        self.state = np.dot(self.F, self.state)

        # Covariance prediction: P = F * P * F^T + Q
        self.covariance = np.dot(np.dot(self.F, self.covariance), self.F.T) + self.Q

    def update(self, measurement: np.ndarray) -> None:
        """Update step using visual position measurements."""
        # Innovation / Residual: y = z - H * x
        z = measurement.reshape(3, 1)
        y = z - np.dot(self.H, self.state)

        # Innovation covariance: S = H * P * H^T + R
        S = np.dot(np.dot(self.H, self.covariance), self.H.T) + self.R

        # Kalman gain: K = P * H^T * S^(-1)
        K = np.dot(np.dot(self.covariance, self.H.T), np.linalg.inv(S))

        # State update: x = x + K * y
        self.state = self.state + np.dot(K, y)

        # Covariance update: P = (I - K * H) * P
        I = np.eye(self.covariance.shape[0])
        self.covariance = np.dot(I - np.dot(K, self.H), self.covariance)

    def process_observation(self, observation: S2ObservationOutput) -> S2ObservationOutput:
        """Processes a single observation through the EKF pipeline."""
        current_time = observation.timestamp
        
        if self.last_timestamp is not None:
            dt = max(current_time - self.last_timestamp, 1e-3)
        else:
            dt = 0.1  # Default time step for the first frame

        self.last_timestamp = current_time

        # 1. Predict state
        self.predict(dt)

        # 2. Extract position measurement from observation
        pos = observation.pose.position
        meas = np.array([pos.x, pos.y, pos.z])

        # 3. Update state with measurement
        self.update(meas)

        # 4. Return updated observation with fused state values
        fused_pos = Position(
            x=float(self.state[0, 0]),
            y=float(self.state[1, 0]),
            z=float(self.state[2, 0]),
        )

        observation.pose.position = fused_pos
        return observation

    def fuse_sequence(self, observations: List[S2ObservationOutput]) -> List[S2ObservationOutput]:
        """Runs the EKF sequentially across an entire list of observation outputs."""
        return [self.process_observation(obs) for obs in observations]