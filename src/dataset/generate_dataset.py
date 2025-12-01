"""Generates synthetic training images of gates from random camera poses."""

import os
import json
import numpy as np
import cv2
import pybullet as p
import pybullet_data
from tqdm import tqdm
import yaml
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


class DatasetGenerator:
    """Creates labelled gate images for detector training."""

    def __init__(self, output_dir="data", headless=True):
        with open('configs/arena_layout.yaml', 'r') as f:
            self.arena = yaml.safe_load(f)
        with open('configs/camera_ov9281_110.yaml', 'r') as f:
            self.camera = yaml.safe_load(f)['camera']

        self.output_dir = output_dir
        self.headless = headless

        self.images_dir = os.path.join(output_dir, "images")
        self.labels_dir = os.path.join(output_dir, "labels")
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)

        self.width = self.camera['resolution']['width']
        self.height = self.camera['resolution']['height']
        self.fov = self.camera['optics']['fov_horizontal']

        print(f"Output: {output_dir}")

    def generate(self, images_per_gate=1000, dist_range=(3.0, 15.0), angle_range_deg=(-60, 60)):
        """Main generation loop - spawns random viewpoints facing each gate."""
        gates = self.arena['gates']['positions']
        total = images_per_gate * len(gates)

        print(f"Generating {total:,} images ({images_per_gate:,}/gate × {len(gates)} gates)")
        print(f"Distance: {dist_range[0]}-{dist_range[1]}m, Angle: ±{angle_range_deg[1]}°\n")

        client = p.connect(p.DIRECT if self.headless else p.GUI)
        if not self.headless:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")

        idx = 0

        for gate_name, gate in tqdm(gates.items(), desc="Gates"):
            # Work out gate centre and yaw
            if gate['type'] == 'single':
                gx, gy = gate['x'], gate['y_center']
                gyaw = 0
            elif 'x1' in gate:
                gx = (gate['x1'] + gate['x2']) / 2
                gy = (gate['y1'] + gate['y2']) / 2
                gyaw = np.deg2rad(gate['yaw'])
            else:
                gx, gy = gate['x'], gate['y_center']
                gyaw = 0

            gz = gate['z']
            label = gate['label']

            for _ in tqdm(range(images_per_gate), desc=f"  {label}", leave=False):
                # Random distance, angle offset, and vertical/lateral jitter
                dist = np.random.uniform(*dist_range)
                ang = np.random.uniform(*np.deg2rad(angle_range_deg))
                lat = np.random.uniform(-1.0, 1.0)
                vert = np.random.uniform(-0.3, 0.3)

                # Position camera facing the gate
                cam_yaw = gyaw + np.pi + ang
                cx = gx - dist * np.cos(gyaw) + lat * np.sin(gyaw)
                cy = gy - dist * np.sin(gyaw) - lat * np.cos(gyaw)
                cz = gz + vert

                cam_pos = [cx, cy, cz]
                cam_orn = p.getQuaternionFromEuler([0, 0, cam_yaw])

                img = self._render(cam_pos, cam_orn)
                if img is None:
                    continue

                cv2.imwrite(os.path.join(self.images_dir, f"{idx:06d}.png"), img)

                meta = {
                    "id": idx,
                    "gate": label,
                    "camera_pos": cam_pos,
                    "gate_pos": [gx, gy, gz],
                    "distance": float(dist)
                }
                with open(os.path.join(self.labels_dir, f"{idx:06d}.json"), 'w') as f:
                    json.dump(meta, f)

                idx += 1

        p.disconnect()
        print(f"\n✓ Generated {idx:,} images in {self.images_dir}")

    # Backward compatibility alias
    generate_dataset = generate

    def _render(self, cam_pos, cam_orn):
        """Renders a single frame and returns mono image."""
        rot = np.array(p.getMatrixFromQuaternion(cam_orn)).reshape(3, 3)
        fwd = rot[:, 0]
        up = rot[:, 2]
        target = np.array(cam_pos) + fwd * 10.0

        view = p.computeViewMatrix(cam_pos, target.tolist(), up.tolist())
        proj = p.computeProjectionMatrixFOV(self.fov, self.width / self.height, 0.05, 100.0)

        _, _, px, _, _ = p.getCameraImage(
            self.width, self.height, view, proj,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        if not isinstance(px, (list, tuple)) or len(px) == 0:
            return None

        rgb = np.array(px, dtype=np.uint8).reshape(self.height, self.width, 4)[:, :, :3]
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate gate detection training data")
    parser.add_argument('--output', type=str, default='data')
    parser.add_argument('--images-per-gate', type=int, default=1000)
    parser.add_argument('--distance-min', type=float, default=3.0)
    parser.add_argument('--distance-max', type=float, default=15.0)
    parser.add_argument('--angle-range', type=float, default=60.0)
    parser.add_argument('--gui', action='store_true', help="Show PyBullet GUI")
    args = parser.parse_args()

    gen = DatasetGenerator(output_dir=args.output, headless=not args.gui)
    gen.generate(
        images_per_gate=args.images_per_gate,
        dist_range=(args.distance_min, args.distance_max),
        angle_range_deg=(-args.angle_range, args.angle_range)
    )


if __name__ == "__main__":
    main()