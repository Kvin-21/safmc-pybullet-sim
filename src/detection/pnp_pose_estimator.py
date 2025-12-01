"""PnP-based gate pose estimation from detected corners."""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict


class GatePoseEstimator:
    """Estimates 3D gate pose from 2D corner detections using PnP."""

    def __init__(
        self,
        gate_width: float = 3.8,
        gate_height: float = 1.9,
        camera_matrix: np.ndarray = None,
        dist_coeffs: np.ndarray = None
    ):
        self.gate_width = gate_width
        self.gate_height = gate_height

        # 3D corners in gate frame (centre origin, X forward, Y right, Z up)
        # Order: TL, TR, BR, BL
        w, h = gate_width / 2, gate_height / 2
        self.gate_corners_3d = np.array([
            [-w,  h, 0],  # top-left
            [ w,  h, 0],  # top-right
            [ w, -h, 0],  # bottom-right
            [-w, -h, 0],  # bottom-left
        ], dtype=np.float32)

        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros(5, dtype=np.float32)

    def estimate_pose(
        self,
        corners_2d: np.ndarray,
        camera_matrix: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None
    ) -> Tuple[bool, np.ndarray, np.ndarray, float]:
        """
        Runs PnP on detected corners. Returns (success, rvec, tvec, reproj_error).
        corners_2d should be shape (4, 2) in order [TL, TR, BR, BL].
        """
        K = camera_matrix if camera_matrix is not None else self.camera_matrix
        D = dist_coeffs if dist_coeffs is not None else self.dist_coeffs

        if K is None:
            raise ValueError("Need a camera matrix")

        pts_2d = corners_2d.reshape(-1, 1, 2).astype(np.float32)
        pts_3d = self.gate_corners_3d.reshape(-1, 1, 3)

        ok, rvec, tvec, _ = cv2.solvePnPRansac(
            objectPoints=pts_3d,
            imagePoints=pts_2d,
            cameraMatrix=K,
            distCoeffs=D,
            reprojectionError=6.0,
            confidence=0.99,
            iterationsCount=100,
            flags=cv2.SOLVEPNP_IPPE
        )

        if not ok or rvec is None or tvec is None:
            return False, None, None, float('inf')

        # Compute reprojection error
        projected, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, D)
        err = np.mean(np.linalg.norm(projected.reshape(-1, 2) - pts_2d.reshape(-1, 2), axis=1))

        return True, rvec.flatten(), tvec.flatten(), err

    def pose_to_matrix(self, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        """Converts rvec/tvec to a 4×4 transformation matrix."""
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec
        return T

    def get_relative_pose(self, rvec: np.ndarray, tvec: np.ndarray) -> Dict[str, np.ndarray]:
        """Extracts useful relative pose info: distance, offsets, yaw error."""
        dist = np.linalg.norm(tvec)
        R, _ = cv2.Rodrigues(rvec)

        # Gate normal direction in camera frame
        gate_normal = R[:, 0]
        yaw_err = np.arctan2(gate_normal[1], gate_normal[0])

        return {
            'position': tvec,
            'distance': dist,
            'lateral_offset': tvec[1],
            'vertical_offset': tvec[2],
            'yaw_error': yaw_err,
            'rotation_matrix': R
        }

    # Backward compatibility aliases
    pose_to_transformation_matrix = pose_to_matrix
    get_gate_relative_position = get_relative_pose