"""Simple downward-facing LiDAR for altitude measurement."""

import numpy as np
import pybullet as p
from typing import Tuple


class DownwardLiDAR:
    """Single-beam LiDAR pointing down to measure height above ground."""

    def __init__(
        self,
        rate_hz: float = 50.0,
        range_max: float = 10.0,
        range_min: float = 0.02,
        noise_sigma: float = 0.01,
        beam_divergence_deg: float = 1.0
    ):
        self.rate_hz = rate_hz
        self.dt = 1.0 / rate_hz
        self.range_max = range_max
        self.range_min = range_min
        self.noise_sigma = noise_sigma
        self.beam_divergence = np.deg2rad(beam_divergence_deg)

        self.last_time = 0.0

    def measure(
        self,
        drone_id: int,
        lidar_link_name: str = "lidar_link",
        current_time: float = 0.0
    ) -> Tuple[float, bool]:
        """
        Fires a ray downward and returns (distance, valid).
        Returns range_max if nothing hit.
        """
        # Rate limit
        if current_time - self.last_time < self.dt:
            return self.range_max, False
        self.last_time = current_time

        link_idx = self._find_link(drone_id, lidar_link_name)
        state = p.getLinkState(drone_id, link_idx, computeForwardKinematics=True)
        pos = state[4]
        orn = state[5]

        # Add slight angular jitter to simulate beam spread
        jx = np.random.normal(0, self.beam_divergence / 3)
        jy = np.random.normal(0, self.beam_divergence / 3)

        direction = np.array([np.tan(jx), np.tan(jy), -1.0])
        direction /= np.linalg.norm(direction)

        rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        direction_world = rot @ direction

        ray_end = tuple(np.array(pos) + direction_world * self.range_max)
        result = p.rayTest(pos, ray_end)

        if result and result[0][0] >= 0:
            hit_frac = result[0][2]
            dist = self.range_max * hit_frac
            dist += np.random.normal(0, self.noise_sigma)
            dist = np.clip(dist, self.range_min, self.range_max)
            return dist, True

        return self.range_max, False

    def _find_link(self, body_id: int, link_name: str) -> int:
        """Looks up link index by name. Returns -1 if not found."""
        for i in range(p.getNumJoints(body_id)):
            info = p.getJointInfo(body_id, i)
            if info[12].decode('utf-8') == link_name:
                return i
        return -1