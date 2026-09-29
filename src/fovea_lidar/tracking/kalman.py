"""Constant-velocity Kalman filter for 3D kinematic object tracking."""

from __future__ import annotations

from typing import Sequence, Tuple
import numpy as np


class ConstantVelocityKalmanFilter:
    """3D Constant-Velocity Linear Kalman Filter.

    State vector: [x, y, z, vx, vy, vz]^T
    Measurement vector: [x, y, z]^T

    Attributes
    ----------
    process_noise_std : float
        Acceleration disturbance noise standard deviation (m/s^2).
    measurement_noise_std : float
        Measurement noise standard deviation (m).
    """

    def __init__(
        self,
        initial_position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        initial_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        process_noise_std: float = 1.0,
        measurement_noise_std: float = 0.15,
        initial_pos_cov: float = 0.5,
        initial_vel_cov: float = 5.0,
    ) -> None:
        self.process_noise_std = float(process_noise_std)
        self.measurement_noise_std = float(measurement_noise_std)

        # State vector: [x, y, z, vx, vy, vz]^T
        self._x = np.array(
            [
                float(initial_position[0]),
                float(initial_position[1]),
                float(initial_position[2]),
                float(initial_velocity[0]),
                float(initial_velocity[1]),
                float(initial_velocity[2]),
            ],
            dtype=np.float64,
        )

        # Error covariance matrix P (6, 6)
        p_diag = [initial_pos_cov**2] * 3 + [initial_vel_cov**2] * 3
        self._P = np.diag(p_diag).astype(np.float64)

        # Measurement matrix H (3, 6)
        self._H = np.zeros((3, 6), dtype=np.float64)
        self._H[:3, :3] = np.eye(3, dtype=np.float64)

        # Measurement noise covariance R (3, 3)
        self._R = np.eye(3, dtype=np.float64) * (self.measurement_noise_std**2)

    @property
    def state(self) -> np.ndarray:
        """Current 6D state vector [x, y, z, vx, vy, vz]."""
        return self._x.copy()

    @property
    def covariance(self) -> np.ndarray:
        """Current 6x6 state estimation error covariance."""
        return self._P.copy()

    @property
    def position(self) -> Tuple[float, float, float]:
        """Estimated 3D position (x, y, z) in meters."""
        return (float(self._x[0]), float(self._x[1]), float(self._x[2]))

    @property
    def velocity(self) -> Tuple[float, float, float]:
        """Estimated 3D velocity (vx, vy, vz) in m/s."""
        return (float(self._x[3]), float(self._x[4]), float(self._x[5]))

    @property
    def speed(self) -> float:
        """Current estimated speed magnitude in m/s."""
        vx, vy, vz = self._x[3], self._x[4], self._x[5]
        return float(np.hypot(np.hypot(vx, vy), vz))

    @property
    def direction(self) -> Tuple[float, float, float]:
        """Normalized 3D unit direction vector (dx, dy, dz)."""
        s = self.speed
        if s > 1e-4:
            return (float(self._x[3] / s), float(self._x[4] / s), float(self._x[5] / s))
        return (1.0, 0.0, 0.0)

    def predict(self, dt: float) -> np.ndarray:
        """Advance the filter state by dt seconds under constant velocity.

        Parameters
        ----------
        dt : float
            Time step elapsed in seconds.

        Returns
        -------
        np.ndarray
            Predicted state vector of shape (6,).
        """
        if dt <= 0.0:
            return self._x.copy()

        # State transition matrix F
        F = np.eye(6, dtype=np.float64)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        # Continuous white noise acceleration process noise covariance Q
        q_var = self.process_noise_std**2
        q11 = (dt**4 / 4.0) * q_var
        q12 = (dt**3 / 2.0) * q_var
        q22 = (dt**2) * q_var

        Q = np.zeros((6, 6), dtype=np.float64)
        for i in range(3):
            Q[i, i] = q11
            Q[i, i + 3] = q12
            Q[i + 3, i] = q12
            Q[i + 3, i + 3] = q22

        # State and covariance prediction
        self._x = F @ self._x
        self._P = F @ self._P @ F.T + Q

        return self._x.copy()

    def update(self, measurement: Sequence[float]) -> np.ndarray:
        """Incorporate a new 3D position measurement (x, y, z).

        Parameters
        ----------
        measurement : Sequence[float]
            Measured position coordinates [x, y, z].

        Returns
        -------
        np.ndarray
            Updated state vector of shape (6,).
        """
        z = np.asarray(measurement[:3], dtype=np.float64)

        # Innovation (residual)
        y = z - self._H @ self._x

        # Innovation covariance
        S = self._H @ self._P @ self._H.T + self._R

        # Kalman gain
        K = self._P @ self._H.T @ np.linalg.inv(S)

        # State update
        self._x = self._x + K @ y

        # Joseph form covariance update for numerical symmetry and positive definiteness
        I = np.eye(6, dtype=np.float64)
        I_KH = I - K @ self._H
        self._P = I_KH @ self._P @ I_KH.T + K @ self._R @ K.T

        return self._x.copy()
