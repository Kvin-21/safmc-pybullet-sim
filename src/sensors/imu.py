"""Simulated 6-DOF IMU with gyro and accelerometer noise models."""

import numpy as np
import pybullet as p
from typing import Tuple


class IMU:
    """Gyroscope + accelerometer with bias drift and white noise."""

    def __init__(
        self,
        rate_hz: float = 1000.0,
        gyro_noise_sigma: float = 0.002,
        accel_noise_sigma: float = 0.02,
        gyro_bias_rw: float = 1e-6,
        accel_bias_rw: float = 1e-5
    ):
        self.rate_hz = rate_hz
        self.dt = 1.0 / rate_hz

        self.gyro_noise = gyro_noise_sigma
        self.accel_noise = accel_noise_sigma
        self.gyro_bias_rw = gyro_bias_rw
        self.accel_bias_rw = accel_bias_rw

        self.gyro_bias = np.zeros(3)
        self.accel_bias = np.zeros(3)

        self.last_time = 0.0
        self.prev_vel = np.zeros(3)

    def measure(self, drone_id: int, current_time: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Grabs angular velocity and linear acceleration in body frame.
        Returns (gyro, accel) both as (3,) arrays.
        """
        pos, quat = p.getBasePositionAndOrientation(drone_id)
        vel, ang_vel = p.getBaseVelocity(drone_id)

        # Gyro is already body-frame from PyBullet
        gyro_true = np.array(ang_vel)

        # Estimate acceleration from velocity change
        if current_time > 0:
            dt = current_time - self.last_time
            if dt > 0:
                accel_world = (np.array(vel) - self.prev_vel) / dt
            else:
                accel_world = np.zeros(3)
        else:
            accel_world = np.zeros(3)

        # Convert to body frame
        rot = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        accel_body = rot.T @ accel_world

        # Subtract gravity (accel measures specific force, not gravity)
        gravity_body = rot.T @ np.array([0, 0, -9.81])
        accel_body -= gravity_body

        # Random walk on biases
        self.gyro_bias += np.random.normal(0, self.gyro_bias_rw * np.sqrt(self.dt), 3)
        self.accel_bias += np.random.normal(0, self.accel_bias_rw * np.sqrt(self.dt), 3)

        # Add bias and white noise
        gyro = gyro_true + self.gyro_bias + np.random.normal(0, self.gyro_noise, 3)
        accel = accel_body + self.accel_bias + np.random.normal(0, self.accel_noise, 3)

        self.last_time = current_time
        self.prev_vel = np.array(vel)

        return gyro, accel

    def reset(self):
        """Clears biases and velocity state."""
        self.gyro_bias = np.zeros(3)
        self.accel_bias = np.zeros(3)
        self.prev_vel = np.zeros(3)
        self.last_time = 0.0