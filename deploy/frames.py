"""Frame conversion between the vision tracker and the training/planner frame.

Tracker (vision_robotic_table_tennis/outputs/ball-trajectory): origin at AprilTag 5,
middle of the robot-side (near) table edge, on the tabletop (z = --table-z, default 0);
+X across the table (robot's right when facing the net), +Y toward the net, +Z up.

Planner / sim frame (g1_sonic_tt): floor-level origin where the robot stands in training,
+X toward the net, +Y left, +Z up; near table edge at x = near_edge_x, tabletop z = top_z.
    x_sim = near_edge_x + y_trk,  y_sim = -x_trk,  z_sim = top_z + (z_trk - table_z_trk)

Verify the axis signs on the real setup (hold the ball at known table points) before use.
"""
from __future__ import annotations

import numpy as np


class TrackerToSim:
    def __init__(self, near_edge_x: float = 0.55, top_z: float = 0.76, table_z_trk: float = 0.0):
        self.near_edge_x, self.top_z, self.table_z_trk = near_edge_x, top_z, table_z_trk
        # rotation part: sim = R @ trk
        self.R = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    def pos(self, p_trk) -> np.ndarray:
        p = self.R @ np.asarray(p_trk, float)
        return p + np.array([self.near_edge_x, 0.0, self.top_z - self.table_z_trk])

    def vel(self, v_trk) -> np.ndarray:
        return self.R @ np.asarray(v_trk, float)

    def yaw(self, yaw_trk: float) -> float:
        """Heading in the tracker frame (0 = +X_trk) -> heading in the sim frame."""
        return float(np.arctan2(np.sin(yaw_trk - np.pi / 2), np.cos(yaw_trk - np.pi / 2)))


def yaw_rot_inv(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
