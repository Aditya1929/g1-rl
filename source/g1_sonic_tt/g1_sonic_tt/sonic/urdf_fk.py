"""Minimal numpy forward kinematics for one URDF chain (no simulator dependency).

Used offline by scripts/build_library.py for the racket-face IK correction.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axis_angle(axis, q):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * K @ K


class UrdfChain:
    """Chain root_link -> tip_link. pose(q_by_name) -> (R, p) of tip in the root frame."""

    def __init__(self, urdf: str | Path, root_link: str, tip_link: str):
        joints = {}
        for j in ET.parse(urdf).getroot().findall("joint"):
            o = j.find("origin")
            xyz = np.fromstring(o.get("xyz", "0 0 0"), sep=" ") if o is not None else np.zeros(3)
            rpy = np.fromstring(o.get("rpy", "0 0 0"), sep=" ") if o is not None else np.zeros(3)
            ax = j.find("axis")
            lim = j.find("limit")
            joints[j.find("child").get("link")] = dict(
                name=j.get("name"), type=j.get("type"), parent=j.find("parent").get("link"),
                R0=_rpy(*rpy), p0=xyz, axis=np.fromstring(ax.get("xyz"), sep=" ") if ax is not None else None,
                limit=(float(lim.get("lower", -np.inf)), float(lim.get("upper", np.inf))) if lim is not None else None)
        chain, link = [], tip_link
        while link != root_link:
            if link not in joints:
                raise ValueError(f"{tip_link} is not below {root_link}")
            chain.append(joints[link])
            link = joints[link]["parent"]
        self.chain = chain[::-1]
        self.joint_names = [j["name"] for j in self.chain if j["type"] in ("revolute", "continuous")]
        self.limits = {j["name"]: j["limit"] for j in self.chain if j["type"] in ("revolute", "continuous")}

    def pose(self, q: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
        R, p = np.eye(3), np.zeros(3)
        for j in self.chain:
            p = p + R @ j["p0"]
            R = R @ j["R0"]
            if j["type"] in ("revolute", "continuous"):
                R = R @ _axis_angle(j["axis"], q[j["name"]])
        return R, p
