# SAFMC 2026 Drone Racing Simulator

PyBullet-based simulation of an F330 drone racing through a 40m × 20m arena with gates.

## Quick Start

```bash
python scripts/visual_simulator.py
```

## Controls

- **L** – cycle lighting
- **B** – cycle gate colours  
- **D** – move drone to next gate
- **R** – reset drone
- **V** – show camera view
- **C** – capture camera view
- **Q** – quit

## Arena

- 40m × 20m × 2.5m
- Central divider at Y=10m (X=8-28m)
- 5 gates: END, START, GATE_1, GATE_2, GATE_3

## Scripts

- `scripts/visual_simulator.py` – interactive 3D visualiser
- `scripts/validate_arena.py` – checks arena config against spec
- `src/dataset/generate_dataset.py` – generates training images

## Sensors

- OV9281-110 camera (1280×800, 110° FOV, mono)
- 6-DOF IMU with noise model
- Downward LiDAR for altitude

## Hardware Model

- F330 frame with 2212 motors
- Betaflight-style rate controller
- First-order motor dynamics