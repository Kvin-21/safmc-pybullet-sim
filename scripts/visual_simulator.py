"""
Interactive 3D visualiser for the SAFMC 2026 arena and F330 drone.
Use mouse to look around, keyboard for controls.
"""

import pybullet as p
import pybullet_data
import time
import yaml
import numpy as np
import sys
import os
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


class VisualSimulator:
    """Interactive PyBullet viewer for testing the arena layout."""

    def __init__(self):
        with open('configs/arena_layout.yaml', 'r') as f:
            self.arena_cfg = yaml.safe_load(f)
        with open('configs/camera_ov9281_110.yaml', 'r') as f:
            self.cam_cfg = yaml.safe_load(f)

        # Fire up PyBullet with a nice big window
        self.client = p.connect(p.GUI, options="--width=1920 --height=1080")
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 1)

        p.resetDebugVisualizerCamera(
            cameraDistance=35, cameraYaw=50, cameraPitch=-40,
            cameraTargetPosition=[20, 10, 0]
        )

        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1./240.)
        p.setRealTimeSimulation(0)

        self.plane_id = None
        self.wall_ids = []
        self.gate_ids = []
        self.gate_info = []
        self.drone_id = None

        self._build_arena()
        self._build_gates()
        self._load_drone()

        self._print_controls()

        self.cam_window_on = False
        self.last_cam_img = None

    def _print_controls(self):
        print("=" * 70)
        print("SAFMC 2026 Visual Simulator")
        print("=" * 70)
        print("\nMouse: drag to rotate, middle-drag to pan, scroll to zoom")
        print("\nKeys:")
        print("  L - cycle lighting")
        print("  B - cycle gate colours")
        print("  D - move drone to next gate")
        print("  R - reset drone")
        print("  C - capture camera view")
        print("  V - toggle live camera window")
        print("  Q - quit")
        print("=" * 70)

    def _build_arena(self):
        """Creates ground, walls, and the central divider."""
        arena = self.arena_cfg['arena']
        divider = self.arena_cfg['central_divider']

        length = arena['dimensions']['length']
        width = arena['dimensions']['width']
        height = arena['dimensions']['height']

        self.plane_id = p.loadURDF("plane.urdf")
        p.changeVisualShape(self.plane_id, -1, rgbaColor=[0.75, 0.75, 0.75, 1.0])

        # Grid lines every 5m
        for x in range(0, int(length) + 1, 5):
            p.addUserDebugLine([x, 0, 0.001], [x, width, 0.001], [0.4, 0.4, 0.4], 1.5)
        for y in range(0, int(width) + 1, 5):
            p.addUserDebugLine([0, y, 0.001], [length, y, 0.001], [0.4, 0.4, 0.4], 1.5)

        # Boundary walls (semi-transparent netting)
        wall_colour = [0.7, 0.7, 0.8, 0.5]
        walls = [
            ([length/2, 0, height/2], [length/2, 0.05, height/2]),
            ([length/2, width, height/2], [length/2, 0.05, height/2]),
            ([0, width/2, height/2], [0.05, width/2, height/2]),
            ([length, width/2, height/2], [0.05, width/2, height/2]),
        ]
        for pos, half in walls:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=wall_colour)
            self.wall_ids.append(p.createMultiBody(0, col, vis, pos))

        # Central divider
        if divider['enabled']:
            div_len = divider['end_x'] - divider['start_x']
            div_pos = [(divider['start_x'] + divider['end_x'])/2, divider['y'], divider['height']/2]
            div_half = [div_len/2, divider['thickness']/2, divider['height']/2]
            wood = [0.55, 0.35, 0.15, 1.0]

            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=div_half)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=div_half, rgbaColor=wood)
            self.wall_ids.append(p.createMultiBody(0, col, vis, div_pos))

            p.addUserDebugLine(
                [divider['start_x'], divider['y'], 0],
                [divider['end_x'], divider['y'], divider['height']],
                [0.9, 0.6, 0.3], 3
            )

        print(f"✓ Arena: {length}m × {width}m × {height}m")
        print(f"✓ Divider: X={divider['start_x']}-{divider['end_x']}m at Y={divider['y']}m")

    def _build_gates(self):
        """Constructs all the racing gates."""
        gates = self.arena_cfg['gates']
        positions = gates['positions']

        blue = [0.0, 0.4, 0.95, 1.0]
        wood = [0.55, 0.35, 0.15, 1.0]

        print("\n✓ Gates:")

        for name, data in positions.items():
            if data['type'] == 'single':
                parts = self._make_single_gate(data, blue, wood)
                self.gate_ids.extend(parts)
                self.gate_info.append({
                    'label': data['label'],
                    'x': data['x'], 'y': data['y_center'], 'z': data['z'],
                    'yaw': data['yaw']
                })
                text_pos = [data['x'], data['y_center'], data['z'] + 1.8]
                p.addUserDebugText(data['label'], text_pos, textSize=1.5,
                                   textColorRGB=[1, 0.9, 0], lifeTime=0)
                print(f"  {data['label']}: ({data['x']}, {data['y_center']})")

            elif data['type'] == 'double':
                if 'x1' in data:
                    parts = self._make_angled_double(data, blue, wood)
                    xc = (data['x1'] + data['x2']) / 2
                    yc = (data['y1'] + data['y2']) / 2
                else:
                    parts = self._make_double_gate(data, blue, wood)
                    xc, yc = data['x'], data['y_center']

                self.gate_ids.extend(parts)
                self.gate_info.append({
                    'label': data['label'],
                    'x': xc, 'y': yc, 'z': data['z'],
                    'yaw': np.deg2rad(data.get('yaw', 0))
                })
                text_pos = [xc, yc, data['z'] + 1.8]
                p.addUserDebugText(data['label'], text_pos, textSize=1.5,
                                   textColorRGB=[1, 0.9, 0], lifeTime=0)
                print(f"  {data['label']}: ({xc:.1f}, {yc:.1f}) yaw={data.get('yaw', 0)}°")

    def _make_single_gate(self, data, blue, wood):
        """Builds a single-aperture gate (START/END)."""
        parts = []
        x, y = data['x'], data['y_center']

        spec = self.arena_cfg['gates']['single_gate']
        leg_h = spec['leg_height']
        sq_h = spec['square_height']
        outer = spec['outer_size']
        border = spec['border']

        # Two legs
        for offset in [-outer/2, outer/2]:
            pos = [x, y + offset, leg_h/2]
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, leg_h/2])
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, leg_h/2], rgbaColor=wood)
            parts.append(p.createMultiBody(0, col, vis, pos))

        # Frame bars
        fz = leg_h + sq_h/2
        bars = [
            ([x, y, fz + sq_h/2 - border/2], [0.05, outer/2, border/2]),     # top
            ([x, y, fz - sq_h/2 + border/2], [0.05, outer/2, border/2]),     # bottom
            ([x, y - outer/2 + border/2, fz], [0.05, border/2, sq_h/2]),     # left
            ([x, y + outer/2 - border/2, fz], [0.05, border/2, sq_h/2]),     # right
        ]
        for pos, half in bars:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=blue)
            parts.append(p.createMultiBody(0, col, vis, pos))

        return parts

    def _make_double_gate(self, data, blue, wood):
        """Builds a double-aperture gate (axis-aligned)."""
        parts = []
        x, y = data['x'], data['y_center']

        spec = self.arena_cfg['gates']['double_gate']
        leg_h = spec['leg_height']
        sq_h = spec['square_height']
        outer = spec['outer_size']
        border = spec['border']
        total_w = spec['total_width']

        # Four legs
        for offset in [-total_w/2, -outer/2, outer/2, total_w/2]:
            pos = [x, y + offset, leg_h/2]
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, leg_h/2])
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, leg_h/2], rgbaColor=wood)
            parts.append(p.createMultiBody(0, col, vis, pos))

        # Frame bars
        fz = leg_h + sq_h/2
        bars = [
            ([x, y, fz + sq_h/2 - border/2], [0.05, total_w/2, border/2]),     # top
            ([x, y, fz - sq_h/2 + border/2], [0.05, total_w/2, border/2]),     # bottom
            ([x, y - total_w/2 + border/2, fz], [0.05, border/2, sq_h/2]),     # left edge
            ([x, y + total_w/2 - border/2, fz], [0.05, border/2, sq_h/2]),     # right edge
            ([x, y, fz], [0.05, border, sq_h/2]),                              # centre divider
        ]
        for pos, half in bars:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=blue)
            parts.append(p.createMultiBody(0, col, vis, pos))

        return parts

    def _make_angled_double(self, data, blue, wood):
        """Builds a double gate at an angle (Gate 3)."""
        parts = []
        xc = (data['x1'] + data['x2']) / 2
        yc = (data['y1'] + data['y2']) / 2

        angle = np.deg2rad(data['yaw'])
        quat = p.getQuaternionFromEuler([0, 0, angle])
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        spec = self.arena_cfg['gates']['double_gate']
        leg_h = spec['leg_height']
        sq_h = spec['square_height']
        outer = spec['outer_size']
        border = spec['border']
        total_w = spec['total_width']

        # Four legs at rotated positions
        for offset in [-total_w/2, -outer/2, outer/2, total_w/2]:
            dx = -offset * sin_a
            dy = offset * cos_a
            pos = [xc + dx, yc + dy, leg_h/2]
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, leg_h/2])
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, leg_h/2], rgbaColor=wood)
            parts.append(p.createMultiBody(0, col, vis, pos))

        # Frame bars (local coords then rotated)
        fz = leg_h + sq_h/2
        bars = [
            (0, 0, sq_h/2 - border/2, [0.05, total_w/2, border/2]),       # top
            (0, 0, -sq_h/2 + border/2, [0.05, total_w/2, border/2]),      # bottom
            (0, -total_w/2 + border/2, 0, [0.05, border/2, sq_h/2]),      # left
            (0, total_w/2 - border/2, 0, [0.05, border/2, sq_h/2]),       # right
            (0, 0, 0, [0.05, border, sq_h/2]),                            # centre
        ]
        for lx, ly, lz, half in bars:
            dx = -ly * sin_a
            dy = ly * cos_a
            pos = [xc + dx, yc + dy, fz + lz]
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=blue)
            parts.append(p.createMultiBody(0, col, vis, pos, quat))

        return parts

    def _load_drone(self):
        """Spawns the F330 drone between END and START."""
        start = [12.0, 15.0, 1.0]
        orn = p.getQuaternionFromEuler([0, 0, 0])
        self.drone_id = p.loadURDF("assets/urdf/f330.urdf", start, orn,
                                    flags=p.URDF_USE_INERTIA_FROM_FILE)
        print("✓ Drone loaded")

    # --- Interactive controls ---

    def cycle_lighting(self):
        """Steps through various lighting presets."""
        modes = [
            ("Bright", 1, [0.85, 0.85, 0.85, 1.0]),
            ("Standard", 1, [0.75, 0.75, 0.75, 1.0]),
            ("Dim", 1, [0.65, 0.65, 0.65, 1.0]),
            ("Low", 0, [0.50, 0.50, 0.50, 1.0]),
            ("Warm", 1, [0.80, 0.70, 0.60, 1.0]),
            ("Cool", 1, [0.70, 0.75, 0.85, 1.0]),
            ("Very Bright", 1, [0.95, 0.95, 0.95, 1.0]),
            ("Very Dark", 0, [0.35, 0.35, 0.35, 1.0]),
        ]

        if not hasattr(self, '_light_idx'):
            self._light_idx = 0
        self._light_idx = (self._light_idx + 1) % len(modes)

        name, shadows, colour = modes[self._light_idx]
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, shadows)
        p.changeVisualShape(self.plane_id, -1, rgbaColor=colour)

        for wid in self.wall_ids[:-1]:
            wall_col = [c * 0.9 for c in colour[:3]] + [0.5]
            p.changeVisualShape(wid, -1, rgbaColor=wall_col)

        print(f"→ Lighting: {name}")

    def cycle_gate_colours(self):
        """Cycles through different blue shades for gates."""
        blues = [
            ([0.0, 0.40, 0.95, 1.0], "Standard"),
            ([0.0, 0.25, 0.65, 1.0], "Navy"),
            ([0.15, 0.55, 1.0, 1.0], "Sky"),
            ([0.0, 0.50, 0.85, 1.0], "Cyan"),
            ([0.10, 0.20, 0.60, 1.0], "Deep"),
            ([0.0, 0.35, 0.80, 1.0], "Royal"),
            ([0.05, 0.48, 0.98, 1.0], "Electric"),
            ([0.0, 0.30, 0.70, 1.0], "Medium"),
            ([0.20, 0.45, 0.90, 1.0], "Azure"),
            ([0.0, 0.55, 0.75, 1.0], "Teal"),
        ]

        if not hasattr(self, '_blue_idx'):
            self._blue_idx = 0
        self._blue_idx = (self._blue_idx + 1) % len(blues)

        colour, name = blues[self._blue_idx]
        for gid in self.gate_ids:
            try:
                p.changeVisualShape(gid, -1, rgbaColor=colour)
            except:
                pass

        print(f"→ Gate colour: {name}")

    def move_drone_next(self):
        """Teleports drone to approach the next gate in sequence."""
        if not hasattr(self, '_gate_seq_idx'):
            self._gate_seq_idx = 0

        sequence = ['END', 'START', 'GATE_1', 'GATE_2', 'GATE_3']
        label = sequence[self._gate_seq_idx]

        gate = next((g for g in self.gate_info if g['label'] == label), None)
        if not gate:
            return

        dist = 2.5
        yaw = gate['yaw']
        if yaw == 0:
            pos = [gate['x'] - dist, gate['y'], gate['z']]
        else:
            pos = [gate['x'] - dist * np.cos(yaw),
                   gate['y'] - dist * np.sin(yaw),
                   gate['z']]

        orn = p.getQuaternionFromEuler([0, 0, yaw])
        p.resetBasePositionAndOrientation(self.drone_id, pos, orn)
        p.resetBaseVelocity(self.drone_id, [0, 0, 0], [0, 0, 0])

        print(f"→ Drone at {label}")
        self._gate_seq_idx = (self._gate_seq_idx + 1) % len(sequence)

    def reset_drone(self):
        """Puts drone back at the starting position."""
        start = [12.0, 15.0, 1.0]
        orn = p.getQuaternionFromEuler([0, 0, 0])
        p.resetBasePositionAndOrientation(self.drone_id, start, orn)
        p.resetBaseVelocity(self.drone_id, [0, 0, 0], [0, 0, 0])
        self._gate_seq_idx = 0
        print("→ Drone reset")

    def capture_camera(self):
        """Grabs a frame from the drone's onboard camera."""
        # Try to find the camera link, fall back to offset from base
        cam_idx = -1
        for i in range(p.getNumJoints(self.drone_id)):
            info = p.getJointInfo(self.drone_id, i)
            if 'camera' in info[12].decode('utf-8').lower():
                cam_idx = i
                break
        if cam_idx == -1:
            cam_idx = 6

        try:
            state = p.getLinkState(self.drone_id, cam_idx, computeForwardKinematics=True)
            cam_pos, cam_orn = state[4], state[5]
        except:
            pos, orn = p.getBasePositionAndOrientation(self.drone_id)
            cam_pos = [pos[0] + 0.06, pos[1], pos[2] + 0.02]
            cam_orn = orn

        rot = np.array(p.getMatrixFromQuaternion(cam_orn)).reshape(3, 3)
        fwd, up = rot[:, 0], rot[:, 2]
        target = np.array(cam_pos) + fwd * 10.0

        view = p.computeViewMatrix(cam_pos, target.tolist(), up.tolist())

        w = self.cam_cfg['camera']['resolution']['width']
        h = self.cam_cfg['camera']['resolution']['height']
        fov = self.cam_cfg['camera']['optics']['fov_horizontal']
        proj = p.computeProjectionMatrixFOV(fov, w/h, 0.05, 100.0)

        _, _, px, _, _ = p.getCameraImage(w, h, view, proj,
                                           renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgb = np.array(px, dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
        self.last_cam_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        print(f"→ Camera captured ({w}×{h})")
        return self.last_cam_img

    def toggle_cam_window(self):
        """Toggles the live camera feed window."""
        self.cam_window_on = not self.cam_window_on
        if self.cam_window_on:
            print("→ Camera window on")
        else:
            print("→ Camera window off")
            cv2.destroyAllWindows()

    # --- Main loop ---

    def run(self):
        """Runs the interactive simulation loop."""
        print("\nSimulator running...\n")

        mass = p.getDynamicsInfo(self.drone_id, -1)[0]
        hover = mass * 9.81 / 4.0

        motor_pos = [
            [0.1237, -0.1237, 0],
            [0.1237,  0.1237, 0],
            [-0.1237, 0.1237, 0],
            [-0.1237, -0.1237, 0],
        ]

        frame = 0
        try:
            while True:
                keys = p.getKeyboardEvents()

                if ord('l') in keys and keys[ord('l')] & p.KEY_WAS_TRIGGERED:
                    self.cycle_lighting()
                if ord('b') in keys and keys[ord('b')] & p.KEY_WAS_TRIGGERED:
                    self.cycle_gate_colours()
                if ord('d') in keys and keys[ord('d')] & p.KEY_WAS_TRIGGERED:
                    self.move_drone_next()
                if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
                    self.reset_drone()
                if ord('c') in keys and keys[ord('c')] & p.KEY_WAS_TRIGGERED:
                    self.capture_camera()
                if ord('v') in keys and keys[ord('v')] & p.KEY_WAS_TRIGGERED:
                    self.toggle_cam_window()
                if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
                    print("→ Quitting...")
                    break

                # Simple hover to keep drone in place
                for mpos in motor_pos:
                    p.applyExternalForce(self.drone_id, -1, [0, 0, hover], mpos, p.LINK_FRAME)

                p.stepSimulation()

                # Update live camera window every 10 frames
                if self.cam_window_on and frame % 10 == 0:
                    img = self.capture_camera()
                    if img is not None:
                        cv2.imshow("Camera", cv2.resize(img, (1280, 800)))
                        cv2.waitKey(1)

                time.sleep(1./240.)
                frame += 1

        except KeyboardInterrupt:
            print("\n→ Interrupted")
        finally:
            cv2.destroyAllWindows()
            p.disconnect()
            print("→ Closed")


if __name__ == "__main__":
    try:
        sim = VisualSimulator()
        sim.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()