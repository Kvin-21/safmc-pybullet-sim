"""
Gymnasium environment for F330 drone racing in the SAFMC 2026 arena.
Uses Betaflight-style rate control and realistic motor dynamics.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import yaml
from typing import Dict, Tuple, Optional, List


class SAFMCF330Env(gym.Env):
    """
    Gym env for training a vision-based racing drone.
    Obs: 19-dim state vector. Action: thrust + body rates.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    # F330 airframe constants
    MASS = 0.87            # kg
    ARM_LEN = 0.175        # metres
    MOTOR_TAU = 0.03       # motor lag time constant (s)
    KF = 1.2e-5            # thrust coeff N/(rad/s)^2
    KM = 1.5e-7            # moment coeff Nm/(rad/s)^2
    DRAG = np.array([0.8, 0.8, 0.3])  # body-frame drag coeffs

    # Rate controller gains (Betaflight-ish)
    RATE_P = np.array([0.12, 0.12, 0.10])
    RATE_D = np.array([0.004, 0.004, 0.003])

    def __init__(
        self,
        config_path: str = "configs/training_config.yaml",
        arena_config_path: str = "configs/arena_layout.yaml",
        camera_config_path: str = "configs/camera_ov9281_110.yaml",
        render_mode: Optional[str] = None,
        gui: bool = False
    ):
        super().__init__()

        with open(config_path, 'r') as f:
            self.train_cfg = yaml.safe_load(f)
        with open(arena_config_path, 'r') as f:
            self.arena_cfg = yaml.safe_load(f)
        with open(camera_config_path, 'r') as f:
            self.cam_cfg = yaml.safe_load(f)

        self.render_mode = render_mode
        self.gui = gui

        env_cfg = self.train_cfg['training']['environment']
        self.physics_dt = env_cfg['physics_timestep']
        self.control_dt = env_cfg['control_timestep']
        self.steps_per_ctrl = int(self.control_dt / self.physics_dt)
        self.max_steps = env_cfg['max_episode_steps']

        act_lim = self.train_cfg['training']['action_space']['limits']
        self.max_roll = act_lim['roll_rate_max']
        self.max_pitch = act_lim['pitch_rate_max']
        self.max_yaw = act_lim['yaw_rate_max']

        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        obs_dim = self.train_cfg['training']['observation_space']['total_dim']
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.rewards = self.train_cfg['training']['reward_shaping']
        self.domain_rand = self.train_cfg['training']['domain_randomisation']

        # State tracking
        self.step_count = 0
        self.episode_count = 0
        self.current_gate_idx = 0
        self.gates_passed = []
        self.crashed = False
        self.motor_omega = np.zeros(4)
        self.prev_action = np.zeros(4)
        self.C_d = self.DRAG.copy()

        # PyBullet handles
        self.client = None
        self.drone_id = None
        self.plane_id = None
        self.gate_ids = []
        self.arena_ids = []
        self.gate_positions = []

        self._init_pybullet()

    def _init_pybullet(self):
        """Starts physics engine and builds the arena."""
        if self.gui:
            self.client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        else:
            self.client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.physics_dt)
        p.setRealTimeSimulation(0)

        self.plane_id = p.loadURDF("plane.urdf")
        self._build_arena()
        self._build_gates()

    def _build_arena(self):
        """Creates boundary walls and the central divider."""
        arena = self.arena_cfg['arena']
        div = self.arena_cfg['central_divider']

        length = arena['dimensions']['length']
        width = arena['dimensions']['width']
        height = arena['dimensions']['height']
        wall_t = 0.05

        # Boundary walls
        walls = [
            ([length/2, 0, height/2], [length/2, wall_t/2, height/2]),
            ([length/2, width, height/2], [length/2, wall_t/2, height/2]),
            ([0, width/2, height/2], [wall_t/2, width/2, height/2]),
            ([length, width/2, height/2], [wall_t/2, width/2, height/2]),
        ]
        for pos, half in walls:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half,
                                       rgbaColor=[0.3, 0.3, 0.3, 0.3])
            self.arena_ids.append(p.createMultiBody(0, col, vis, pos))

        # Central divider
        if div['enabled']:
            div_len = div['end_x'] - div['start_x']
            div_pos = [(div['start_x'] + div['end_x'])/2, div['centre_y'], div['height']/2]
            div_half = [div_len/2, div['thickness']/2, div['height']/2]
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=div_half)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=div_half,
                                       rgbaColor=[0.6, 0.4, 0.2, 1.0])
            self.arena_ids.append(p.createMultiBody(0, col, vis, div_pos))

    def _build_gates(self):
        """Creates gate obstacles from config."""
        gates = self.arena_cfg['gates']
        positions = gates['positions']
        W = gates['aperture']['width']
        H = gates['aperture']['height']
        T = gates['frame']['thickness']
        D = gates['frame']['depth']

        for _, data in positions.items():
            pos = data['position']
            yaw = np.deg2rad(data['yaw'])

            self.gate_positions.append({
                'position': np.array(pos),
                'yaw': yaw,
                'label': data['label']
            })

            gid = self._make_gate_frame(pos, yaw, W, H, T, D)
            self.gate_ids.append(gid)

    def _make_gate_frame(self, pos: List[float], yaw: float,
                         w: float, h: float, t: float, d: float) -> int:
        """Builds a simplified gate (just top bar for now)."""
        half = [w/2 + t, t/2, d/2]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half,
                                   rgbaColor=[1.0, 0.5, 0.0, 1.0])
        quat = p.getQuaternionFromEuler([0, 0, yaw])

        top_local = np.array([0, 0, h/2 + t/2])
        top_world = pos + self._rotate_z(top_local, yaw)

        return p.createMultiBody(0, col, vis, top_world.tolist(), quat)

    @staticmethod
    def _rotate_z(v: np.ndarray, yaw: float) -> np.ndarray:
        """Rotates a vector about Z by yaw."""
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        return R @ v

    # ---- Gym interface ----

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Resets the drone to start position."""
        super().reset(seed=seed)

        if self.drone_id is not None:
            p.removeBody(self.drone_id)

        start_pos = [1.5, 10.0, 1.0]
        start_orn = p.getQuaternionFromEuler([0, 0, np.pi/2])
        self.drone_id = p.loadURDF("assets/urdf/f330.urdf", start_pos, start_orn,
                                    flags=p.URDF_USE_INERTIA_FROM_FILE)

        if self.domain_rand['enabled']:
            self._randomise()

        self.step_count = 0
        self.current_gate_idx = 0
        self.gates_passed = []
        self.crashed = False
        self.motor_omega = np.zeros(4)
        self.prev_action = np.zeros(4)
        self.episode_count += 1

        return self._observe(), self._info()

    def step(self, action: np.ndarray):
        """Runs one control step (may be multiple physics steps)."""
        action = np.clip(action, self.action_space.low, self.action_space.high)

        thrust = action[0]
        rates = action[1:] * np.array([self.max_roll, self.max_pitch, self.max_yaw])

        for _ in range(self.steps_per_ctrl):
            self._apply_control(thrust, rates)
            p.stepSimulation()

        self.step_count += 1
        obs = self._observe()
        reward = self._reward(obs, action)

        terminated = self._is_done()
        truncated = self.step_count >= self.max_steps

        self.prev_action = action
        return obs, reward, terminated, truncated, self._info()

    def _apply_control(self, thrust_norm: float, rate_cmd: np.ndarray):
        """Betaflight-style rate controller + motor lag."""
        pos, quat = p.getBasePositionAndOrientation(self.drone_id)
        _, ang_vel = p.getBaseVelocity(self.drone_id)
        omega = np.array(ang_vel)

        # PD rate controller
        rate_err = rate_cmd - omega
        torque = self.RATE_P * rate_err - self.RATE_D * omega

        # Total collective thrust
        thrust_total = thrust_norm * 4 * self.MASS * 9.81

        # Base per-motor thrust then add torque corrections
        motor_thrust = np.full(4, thrust_total / 4)

        arm = self.ARM_LEN
        roll_mix = torque[0] / (4 * arm * self.KF * 1e4)
        pitch_mix = torque[1] / (4 * arm * self.KF * 1e4)
        yaw_mix = torque[2] / (4 * self.KM * 1e4)

        # FR, FL, RL, RR
        motor_thrust[0] += roll_mix + pitch_mix - yaw_mix
        motor_thrust[1] += -roll_mix + pitch_mix + yaw_mix
        motor_thrust[2] += -roll_mix - pitch_mix - yaw_mix
        motor_thrust[3] += roll_mix - pitch_mix + yaw_mix

        motor_thrust = np.clip(motor_thrust, 0, thrust_total)

        # First-order motor lag
        omega_cmd = np.sqrt(np.abs(motor_thrust) / self.KF)
        alpha = 1.0 - np.exp(-self.physics_dt / self.MOTOR_TAU)
        self.motor_omega += alpha * (omega_cmd - self.motor_omega)

        thrust_actual = self.KF * self.motor_omega**2

        # Motor positions (X-config)
        arm = self.ARM_LEN
        mpos = np.array([
            [arm, -arm, 0], [arm, arm, 0],
            [-arm, arm, 0], [-arm, -arm, 0]
        ])

        for i in range(4):
            f_world = self._body_to_world([0, 0, thrust_actual[i]], quat)
            p_world = self._body_to_world(mpos[i], quat, pos)
            p.applyExternalForce(self.drone_id, -1, f_world.tolist(),
                                  p_world.tolist(), p.WORLD_FRAME)

        # Aerodynamic drag
        vel, _ = p.getBaseVelocity(self.drone_id)
        vel_body = self._world_to_body(vel, quat)
        drag_body = -self.C_d * vel_body * np.abs(vel_body)
        drag_world = self._body_to_world(drag_body, quat)
        p.applyExternalForce(self.drone_id, -1, drag_world.tolist(), pos, p.WORLD_FRAME)

    def _body_to_world(self, v, quat, origin=(0, 0, 0)):
        R = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        return R @ np.array(v) + np.array(origin)

    def _world_to_body(self, v, quat):
        R = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        return R.T @ np.array(v)

    def _observe(self) -> np.ndarray:
        """Builds the 19-dim observation vector."""
        pos, quat = p.getBasePositionAndOrientation(self.drone_id)
        vel, ang_vel = p.getBaseVelocity(self.drone_id)
        euler = p.getEulerFromQuaternion(quat)

        # Target gate
        if self.current_gate_idx < len(self.gate_positions):
            gate = self.gate_positions[self.current_gate_idx]
        else:
            gate = self.gate_positions[-1]

        gate_rel_world = gate['position'] - np.array(pos)
        gate_rel_body = self._world_to_body(gate_rel_world, quat)

        yaw_err = self._wrap(gate['yaw'] - euler[2])
        visible = (gate_rel_body[0] > 0 and
                   np.linalg.norm(gate_rel_body) < 20.0 and
                   abs(np.arctan2(gate_rel_body[1], gate_rel_body[0])) < np.deg2rad(60))

        vel_body = self._world_to_body(vel, quat)

        # LiDAR altitude
        ray_end = (pos[0], pos[1], pos[2] - 10.0)
        hit = p.rayTest(pos, ray_end)
        alt = 10.0 * hit[0][2] if hit[0][0] >= 0 else 10.0

        return np.concatenate([
            gate_rel_body, [yaw_err], [float(visible)],
            vel_body, euler, ang_vel, [alt], self.prev_action
        ]).astype(np.float32)

    def _reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        """Computes shaped reward."""
        r = 0.0
        gate_rel = obs[0:3]
        visible = obs[4]

        pos, _ = p.getBasePositionAndOrientation(self.drone_id)
        dist = np.linalg.norm(gate_rel)

        if visible > 0.5:
            r += self.rewards['visibility_bonus'] * self.control_dt

        r += self.rewards['distance_penalty'] * dist
        r += self.rewards['lateral_error_penalty'] * np.linalg.norm(gate_rel[1:3])**2
        r += self.rewards['yaw_error_penalty'] * obs[3]**2
        r += self.rewards['action_smoothness'] * np.linalg.norm(action - self.prev_action)

        if self._check_gate_passed():
            r += self.rewards['gate_pass']
            self.current_gate_idx += 1
            self.gates_passed.append(self.step_count)

        if self._check_collision():
            r += self.rewards['collision_penalty']
            self.crashed = True

        dims = self.arena_cfg['arena']['dimensions']
        if pos[0] < 0 or pos[0] > dims['length'] or \
           pos[1] < 0 or pos[1] > dims['width'] or \
           pos[2] > dims['height']:
            r += self.rewards['boundary_penalty']

        if len(self.gates_passed) == len(self.gate_positions):
            r += self.rewards['lap_completion']

        return r

    def _check_gate_passed(self) -> bool:
        if self.current_gate_idx >= len(self.gate_positions):
            return False

        pos, _ = p.getBasePositionAndOrientation(self.drone_id)
        gate = self.gate_positions[self.current_gate_idx]
        rel = np.array(pos) - gate['position']

        yaw = gate['yaw']
        c, s = np.cos(yaw), np.sin(yaw)
        fwd = rel[0] * c + rel[1] * s
        lat = -rel[0] * s + rel[1] * c

        W = self.arena_cfg['gates']['aperture']['width']
        H = self.arena_cfg['gates']['aperture']['height']

        return abs(lat) < W/2 and abs(rel[2]) < H/2 and -0.2 < fwd < 0.5

    def _check_collision(self) -> bool:
        contacts = p.getContactPoints(bodyA=self.drone_id)
        pos, _ = p.getBasePositionAndOrientation(self.drone_id)
        for c in contacts:
            if c[2] != self.plane_id or pos[2] < 0.05:
                return True
        return False

    def _is_done(self) -> bool:
        if self.crashed:
            return True

        pos, _ = p.getBasePositionAndOrientation(self.drone_id)
        dims = self.arena_cfg['arena']['dimensions']

        if pos[0] < -1 or pos[0] > dims['length'] + 1 or \
           pos[1] < -1 or pos[1] > dims['width'] + 1 or \
           pos[2] < 0 or pos[2] > dims['height'] + 0.5:
            return True

        if len(self.gates_passed) == len(self.gate_positions):
            return True

        return False

    def _info(self) -> dict:
        return {
            'step': self.step_count,
            'gates_passed': len(self.gates_passed),
            'current_gate_idx': self.current_gate_idx,
            'collision': self.crashed
        }

    def _randomise(self):
        """Applies domain randomisation."""
        if not self.domain_rand['enabled']:
            return

        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, int(np.random.rand() > 0.3))

        for gid in self.gate_ids:
            col = [np.random.uniform(0.2, 1.0) for _ in range(3)] + [1.0]
            p.changeVisualShape(gid, -1, rgbaColor=col)

        pcol = [np.random.uniform(0.3, 0.7) for _ in range(3)] + [1.0]
        p.changeVisualShape(self.plane_id, -1, rgbaColor=pcol)

        dyn = self.domain_rand['dynamics']
        if self.drone_id is not None:
            scale = np.random.uniform(1 - dyn['mass_variation'], 1 + dyn['mass_variation'])
            p.changeDynamics(self.drone_id, -1, mass=self.MASS * scale)
            drag_scale = np.random.uniform(1 - dyn['drag_variation'], 1 + dyn['drag_variation'])
            self.C_d = self.DRAG * drag_scale

    @staticmethod
    def _wrap(angle: float) -> float:
        """Wraps angle to [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def render(self):
        """Returns RGB array if render_mode is set."""
        if self.render_mode != "rgb_array":
            return None

        pos, _ = p.getBasePositionAndOrientation(self.drone_id)
        p.resetDebugVisualizerCamera(5.0, 50, -35, pos)

        view = p.computeViewMatrixFromYawPitchRoll(pos, 5.0, 50, -35, 0, 2)
        proj = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 100.0)
        _, _, px, _, _ = p.getCameraImage(640, 480, view, proj,
                                           renderer=p.ER_BULLET_HARDWARE_OPENGL)
        return np.array(px, dtype=np.uint8)[:, :, :3]

    def close(self):
        """Disconnects physics engine."""
        if self.client is not None:
            p.disconnect(self.client)