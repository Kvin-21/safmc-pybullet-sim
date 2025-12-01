"""OV9281-110 camera simulation (global shutter, mono, 110° FOV)."""

import numpy as np
import cv2
import pybullet as p
from typing import Tuple, Dict, Optional
import yaml


class OV9281Camera:
    """Simulates the OV9281-110 monochrome global shutter camera."""

    def __init__(self, config_path: str = "configs/camera_ov9281_110.yaml"):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)['camera']

        self.width = cfg['resolution']['width']
        self.height = cfg['resolution']['height']

        self.fx = cfg['intrinsics']['fx']
        self.fy = cfg['intrinsics']['fy']
        self.cx = cfg['intrinsics']['cx']
        self.cy = cfg['intrinsics']['cy']

        self.k1 = cfg['distortion']['k1']
        self.k2 = cfg['distortion']['k2']
        self.p1 = cfg['distortion']['p1']
        self.p2 = cfg['distortion']['p2']

        self.fov_h = cfg['optics']['fov_horizontal']
        self.fov_v = cfg['optics']['fov_vertical']

        self.read_noise = cfg['noise']['read_noise_sigma']
        self.shot_noise_on = cfg['noise']['shot_noise_enabled']

        self.blur_subframes = cfg['processing']['motion_blur_subframes']
        self.vignette_on = cfg['processing']['vignetting_enabled']

        # OpenCV-friendly matrices
        self.K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)

        self.dist_coeffs = np.array([self.k1, self.k2, self.p1, self.p2, 0.0], dtype=np.float32)

        self._vignette_mask = self._make_vignette_mask()

    def render_frame(
        self,
        drone_id: int,
        camera_link_name: str = "camera_link",
        apply_noise: bool = True,
        apply_motion_blur: bool = False,
        domain_randomisation: Optional[Dict] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Grabs a frame from the drone's camera link.
        Returns (mono_image, depth, segmentation, metadata).
        """
        link_idx = self._find_link(drone_id, camera_link_name)
        state = p.getLinkState(drone_id, link_idx, computeForwardKinematics=True)
        cam_pos, cam_orn = state[4], state[5]

        rot = np.array(p.getMatrixFromQuaternion(cam_orn)).reshape(3, 3)
        forward = rot[:, 0]
        up = rot[:, 2]
        target = np.array(cam_pos) + forward * 10.0

        view_mat = p.computeViewMatrix(cam_pos, target.tolist(), up.tolist())
        proj_mat = p.computeProjectionMatrixFOV(
            self.fov_h, self.width / self.height, 0.05, 100.0
        )

        if apply_motion_blur:
            frames = []
            for _ in range(self.blur_subframes):
                _, _, px, depth, seg = p.getCameraImage(
                    self.width, self.height, view_mat, proj_mat,
                    renderer=p.ER_BULLET_HARDWARE_OPENGL,
                    flags=p.ER_NO_SEGMENTATION_MASK
                )
                frames.append(np.array(px, dtype=np.float32)[:, :, :3])
            rgb = np.mean(frames, axis=0).astype(np.uint8)
        else:
            _, _, px, depth, seg = p.getCameraImage(
                self.width, self.height, view_mat, proj_mat,
                renderer=p.ER_BULLET_HARDWARE_OPENGL
            )
            rgb = np.array(px, dtype=np.uint8)[:, :, :3]
            depth = np.array(depth, dtype=np.float32)
            seg = np.array(seg, dtype=np.int32)

        mono = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        if domain_randomisation:
            mono = self._apply_domain_rand(mono, domain_randomisation)
        if apply_noise:
            mono = self._add_noise(mono)
        if self.vignette_on:
            mono = self._apply_vignette(mono)

        meta = {
            'position': cam_pos,
            'orientation': cam_orn,
            'intrinsics': self.K.tolist(),
            'distortion': self.dist_coeffs.tolist(),
            'width': self.width,
            'height': self.height,
            'fov_horizontal': self.fov_h,
            'fov_vertical': self.fov_v
        }

        return mono, depth, seg, meta

    def _find_link(self, body_id: int, link_name: str) -> int:
        for i in range(p.getNumJoints(body_id)):
            info = p.getJointInfo(body_id, i)
            if info[12].decode('utf-8') == link_name:
                return i
        return -1

    def _add_noise(self, img: np.ndarray) -> np.ndarray:
        """Adds read noise and optional shot noise."""
        out = img.astype(np.float32)
        out += np.random.normal(0, self.read_noise, img.shape)

        if self.shot_noise_on:
            sigma = np.sqrt(np.maximum(out, 0))
            out += np.random.normal(0, 1, img.shape) * sigma

        return np.clip(out, 0, 255).astype(np.uint8)

    def _make_vignette_mask(self) -> np.ndarray:
        """Pre-computes a radial brightness falloff mask."""
        y, x = np.ogrid[:self.height, :self.width]
        cx, cy = self.width / 2, self.height / 2
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        r_max = np.sqrt(cx**2 + cy**2)
        r_norm = r / r_max
        return np.cos(r_norm * np.pi / 2) ** 1.5

    def _apply_vignette(self, img: np.ndarray) -> np.ndarray:
        out = img.astype(np.float32) * self._vignette_mask
        return np.clip(out, 0, 255).astype(np.uint8)

    def _apply_domain_rand(self, img: np.ndarray, params: Dict) -> np.ndarray:
        """Randomly adjusts exposure/gain and adds glare spots."""
        out = img.astype(np.float32)

        if 'exposure_jitter' in params:
            scale = np.random.uniform(1 - params['exposure_jitter'], 1 + params['exposure_jitter'])
            out *= scale

        if 'gain_jitter' in params:
            scale = np.random.uniform(1 - params['gain_jitter'], 1 + params['gain_jitter'])
            out *= scale

        if 'glare_prob' in params and np.random.rand() < params['glare_prob']:
            gx = np.random.randint(0, self.width)
            gy = np.random.randint(0, self.height)
            gr = np.random.randint(20, 80)
            yy, xx = np.ogrid[:self.height, :self.width]
            mask = (xx - gx)**2 + (yy - gy)**2 <= gr**2
            out[mask] += np.random.uniform(100, 200)

        return np.clip(out, 0, 255).astype(np.uint8)

    def project_3d_to_2d(self, points_3d: np.ndarray) -> np.ndarray:
        """Projects (N, 3) camera-frame points to (N, 2) pixel coords."""
        x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
        z = np.maximum(z, 1e-6)
        u = self.fx * (x / z) + self.cx
        v = self.fy * (y / z) + self.cy
        return np.column_stack([u, v])

    def undistort_image(self, img: np.ndarray) -> np.ndarray:
        """Removes lens distortion using calibration params."""
        return cv2.undistort(img, self.K, self.dist_coeffs)

    def get_intrinsics(self) -> Dict:
        """Returns camera intrinsics as a dict."""
        return {
            'fx': self.fx, 'fy': self.fy,
            'cx': self.cx, 'cy': self.cy,
            'k1': self.k1, 'k2': self.k2,
            'p1': self.p1, 'p2': self.p2,
            'width': self.width, 'height': self.height
        }