"""Deploy bridge: vision ball tracker -> strike planner -> trained policy -> SONIC tokens.

    vision ball-trajectory (ZMQ 5564, msgpack) ─┐
    G1 rt/lowstate (DDS: IMU + 29 joints)     ──┼─> bridge @ 50 Hz ─> token (ZMQ 5556) ─> gear_sonic_deploy
    robot pose in table frame (static / vision) ┘

Swing logic mirrors scripts/eval_ball.py: when a valid strike is predicted within the
clip's approach time, start the swing clock (frame = hit_frame - steps_to_hit), refine
the racket target until `commit_s` before impact, finish the clip, then idle.
While idle, the library ready-pose token is streamed (as stream_tokens.py does).

Token (obs_spec.json "action", same as the env's SonicTokenAction):
    residual: z = quantize(z_ref[clip, frame] + lambda * clip(policy(obs), -1, 1))
    absolute: z = quantize(tanh(policy(obs)))
The face-normal target is the planned racket velocity direction (HITTER).

SAFETY: dry-run by default (no tokens are published). Validate sim-to-sim in MuJoCo
with the real SONIC deploy controller first. --live publishes to the controller.

  python deploy/bridge.py --policy logs/.../exported/policy.pt            # dry run, prints plan + tokens
  python deploy/bridge.py --policy ... --live --robot-source lowstate      # real robot, supervised
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/g1_sonic_tt"))
sys.path.insert(0, str(ROOT / "deploy"))

from frames import TrackerToSim, yaw_rot_inv  # noqa: E402
from token_stream import TokenPublisher  # noqa: E402

from g1_sonic_tt import paths  # noqa: E402
from g1_sonic_tt.ball.model import BallParams  # noqa: E402
from g1_sonic_tt.planner.strike import PlannerParams, plan  # noqa: E402
from g1_sonic_tt.sonic.constants import MUJOCO_TO_ISAACLAB, SONIC_JOINTS  # noqa: E402
from g1_sonic_tt.sonic.constants import FSQ_MAX, FSQ_MIN  # noqa: E402
from g1_sonic_tt.sonic.fsq import quantize, squash  # noqa: E402
from g1_sonic_tt.sonic.library import SwingLibrary  # noqa: E402

TRACKING_STATES_REJECT = {"lost", "initializing", "waiting_for_3d", "stream_stale", "stationary",
                          "invalid_timestamp", "out_of_order"}


# --------------------------------------------------------------------------- inputs
class TrackerSource:
    """Latest result from vision_robotic_table_tennis/outputs/ball-trajectory/run.py."""

    def __init__(self, addr: str):
        import msgpack
        import zmq

        self.msgpack = msgpack
        self.sock = zmq.Context.instance().socket(zmq.SUB)
        self.sock.setsockopt(zmq.SUBSCRIBE, b"")
        self.sock.setsockopt(zmq.CONFLATE, 1)  # keep only the newest message
        self.sock.connect(addr)
        self.latest = None

    def poll(self):
        import zmq

        try:
            while True:
                self.latest = self.msgpack.unpackb(self.sock.recv(zmq.NOBLOCK), raw=False)
        except zmq.Again:
            pass
        r = self.latest
        if not r or r.get("status") in TRACKING_STATES_REJECT or "position_m" not in r:
            return None
        return r


class LowStateSource:
    """G1 IMU + joints from rt/lowstate (unitree_sdk2py). Motor order = MuJoCo joint order."""

    def __init__(self, iface: str):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

        ChannelFactoryInitialize(0, iface)
        self.msg = None
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self._cb, 10)

    def _cb(self, msg):
        self.msg = msg

    def read(self):
        m = self.msg
        if m is None:
            return None
        q_mj = np.array([m.motor_state[i].q for i in range(29)])
        dq_mj = np.array([m.motor_state[i].dq for i in range(29)])
        w, x, y, z = m.imu_state.quaternion
        return dict(q_sonic=q_mj[MUJOCO_TO_ISAACLAB], dq_sonic=dq_mj[MUJOCO_TO_ISAACLAB],
                    ang_vel_b=np.array(m.imu_state.gyroscope), quat_wxyz=np.array([w, x, y, z]))


class NominalRobotSource:
    """Dry-run stand-in: robot holding SONIC's default pose, upright."""

    def __init__(self, default_sonic: np.ndarray):
        self.q = default_sonic

    def read(self):
        return dict(q_sonic=self.q, dq_sonic=np.zeros(29), ang_vel_b=np.zeros(3), quat_wxyz=np.array([1.0, 0, 0, 0]))


def gravity_b(quat_wxyz):
    w, x, y, z = quat_wxyz
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return R.T @ np.array([0.0, 0.0, -1.0])


# --------------------------------------------------------------------------- obs
class ObsBuilder:
    """Rebuilds the actor observation exactly as the PLAY env (no noise), from obs_spec.json."""

    def __init__(self, spec_path: Path):
        spec = json.loads(spec_path.read_text())
        self.terms = [(n, int(np.prod(d))) for n, d in spec["terms"]]
        self.sim_joints = spec["joint_names"]
        self.default_sim = np.array(spec["default_joint_pos"])
        self.sonic_to_sim = [SONIC_JOINTS.index(n) for n in self.sim_joints]
        act = spec.get("action", {"mode": "absolute", "residual_scale": 0.0})  # pre-residual exports
        self.mode, self.residual_scale = act["mode"], float(act["residual_scale"])

    def token(self, a: torch.Tensor, z_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Policy output -> (quantized token, residual dz) exactly as SonicTokenAction."""
        if self.mode == "residual":
            dz = a.clamp(-1.0, 1.0)
            z = torch.clamp(torch.as_tensor(z_ref)[None] + self.residual_scale * dz, FSQ_MIN, FSQ_MAX)
            return quantize(z)[0].numpy(), dz[0].numpy()
        if self.mode == "absolute":
            z = squash(a)
            return quantize(z)[0].numpy(), np.zeros(64, np.float32)
        raise ValueError(self.mode)

    def build(self, rs, target_pos_b, target_vel_b, t_strike, z_ref, prev_residual, prev_token,
              target_normal_b=None, base_err_b=(0.0, 0.0)):
        q_sim = rs["q_sonic"][self.sonic_to_sim]
        dq_sim = rs["dq_sonic"][self.sonic_to_sim]
        g = gravity_b(rs["quat_wxyz"])
        if target_normal_b is None:
            target_normal_b = np.asarray(target_vel_b) / max(np.linalg.norm(target_vel_b), 1e-6)
        vals = {
            "base_ang_vel": rs["ang_vel_b"], "projected_gravity": g, "forward_vec": np.array([1.0, 0.0]),
            "base_target_err": np.asarray(base_err_b), "racket_target_pos": target_pos_b,
            "racket_target_vel": target_vel_b, "racket_target_normal": target_normal_b,
            "t_strike": np.array([t_strike]), "ref_token": z_ref, "prev_residual": prev_residual,
            "swing_type": np.array([1.0, 0.0]), "joint_pos": q_sim - self.default_sim, "joint_vel": dq_sim,
            "prev_token": prev_token,
        }
        parts = []
        for name, dim in self.terms:
            x = np.asarray(vals[name], dtype=np.float32).reshape(-1)
            assert x.size == dim, f"{name}: {x.size} != {dim}"
            parts.append(x)
        return torch.from_numpy(np.concatenate(parts))[None]


# --------------------------------------------------------------------------- main loop
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, help="exported/policy.pt from scripts/play.py")
    ap.add_argument("--spec", default=None, help="obs_spec.json (default: next to the policy)")
    ap.add_argument("--tracker", default="tcp://127.0.0.1:5564")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--live", action="store_true", help="publish tokens to the SONIC controller")
    ap.add_argument("--robot-source", choices=["nominal", "lowstate"], default="nominal")
    ap.add_argument("--iface", default="eth0")
    ap.add_argument("--robot-x", type=float, default=0.0, help="pelvis x in the sim/planner frame (m)")
    ap.add_argument("--robot-y", type=float, default=0.0)
    ap.add_argument("--robot-yaw", type=float, default=0.0)
    ap.add_argument("--pelvis-height", type=float, default=0.78)
    ap.add_argument("--latency", type=float, default=0.05, help="vision latency to extrapolate over (s)")
    ap.add_argument("--commit-s", type=float, default=0.3)
    ap.add_argument("--table-z-trk", type=float, default=0.0)
    args = ap.parse_args()

    spec = Path(args.spec) if args.spec else Path(args.policy).with_name("obs_spec.json")
    obsb = ObsBuilder(spec)
    policy = torch.jit.load(args.policy).eval()
    lib = SwingLibrary(paths.FOREHAND_LIBRARY)
    hit_frame, T, dt = lib.hit_frame, lib.num_frames, 1.0 / lib.fps
    idle_token = lib.tokens[0, -1].numpy()  # ready pose (clip end == start)

    conv = TrackerToSim(table_z_trk=args.table_z_trk)
    prm = PlannerParams(ball=BallParams())
    tracker = TrackerSource(args.tracker)
    from g1_sonic_tt.sonic.decoder import load_decoder_params
    robot = LowStateSource(args.iface) if args.robot_source == "lowstate" else \
        NominalRobotSource(load_decoder_params(paths.DECODER_PARAMS_JSON)[0].numpy())
    pub = TokenPublisher(args.port, live=args.live)
    print(f"[bridge] {'LIVE' if args.live else 'DRY RUN'}; tracker {args.tracker}; robot {args.robot_source}; "
          f"action {obsb.mode} (lambda {obsb.residual_scale})")

    root = np.array([args.robot_x, args.robot_y, args.pelvis_height])
    Rinv = yaw_rot_inv(args.robot_yaw)
    swinging, frame, motion = False, 0, 0
    tgt_p, tgt_v = np.zeros(3), np.zeros(3)
    prev_token = idle_token.copy()
    prev_dz = np.zeros(64, np.float32)
    next_t = time.monotonic()
    while True:
        next_t += dt
        r = tracker.poll()
        rs = robot.read()
        if rs is None:
            pub.send(idle_token)
            time.sleep(max(0.0, next_t - time.monotonic()))
            continue

        if r is not None and (not swinging or frame < hit_frame):
            p = torch.tensor(conv.pos(r["position_m"]), dtype=torch.float32)[None]
            v = torch.tensor(conv.vel(r["velocity_m_s"]), dtype=torch.float32)[None]
            p = p + v * args.latency  # crude latency compensation (drag/bounce ignored over 50 ms)
            ok, t_hit, p_hit, v_r, _ = plan(p, v, prm)
            if bool(ok[0]):
                t_rem = float(t_hit[0]) - args.latency
                pos_b, vel_b = Rinv @ (p_hit[0].numpy() - root), Rinv @ v_r[0].numpy()
                steps = int(round(t_rem / dt))
                if not swinging and 0 < steps <= hit_frame:
                    swinging, frame = True, hit_frame - steps
                    motion = int(lib.nearest(torch.tensor(pos_b, dtype=torch.float32)[None])[0])
                    tgt_p, tgt_v = pos_b, vel_b
                    print(f"[bridge] swing: impact in {t_rem:.2f}s at {np.round(pos_b, 3)} (clip {lib.motion_keys[motion]})")
                elif swinging and t_rem > args.commit_s:
                    tgt_p, tgt_v = pos_b, vel_b

        if swinging:
            z_ref = lib.tokens[motion, min(frame, T - 1)].numpy()
            obs = obsb.build(rs, tgt_p, tgt_v, (hit_frame - frame) * dt, z_ref, prev_dz, prev_token)
            with torch.no_grad():
                token, prev_dz = obsb.token(policy(obs), z_ref)
            frame += 1
            if frame >= T:
                swinging = False
        else:
            token, prev_dz = idle_token, np.zeros(64, np.float32)
        pub.send(token)
        prev_token = token
        time.sleep(max(0.0, next_t - time.monotonic()))


if __name__ == "__main__":
    main()
