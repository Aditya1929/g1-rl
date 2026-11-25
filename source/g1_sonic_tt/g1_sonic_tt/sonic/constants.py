"""SONIC G1 conventions shared by the decoder, the env and the deploy bridge.

Sources:
  * Joint order: sonic-dataset/build_dataset.py (checked against
    gear_sonic_deploy policy_parameters.hpp mujoco_to_isaaclab).
  * Default angles / action scale: sonic-dataset/data/decoder_params.json
    (extracted from policy_parameters.hpp), MuJoCo order -> remapped here.
  * Decoder input layout: gear_sonic_deploy/policy/release/observation_config.yaml.
"""
from __future__ import annotations

# SONIC / IsaacLab joint order. Decoder inputs and outputs use this order.
SONIC_JOINTS = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]
MUJOCO_TO_ISAACLAB = [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
                      16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]
NUM_JOINTS = 29

# Token: 2 FSQ tokens x 32 dims. vector_quantize_pytorch FSQ with 32 levels per dim:
# values k/16, k in [-16, 15] -> [-1, 0.9375] (verified by probing the release encoder ONNX).
TOKEN_DIM = 64
FSQ_STEP = 1.0 / 16.0
FSQ_MIN = -1.0
FSQ_MAX = 15.0 / 16.0

# Decoder input (994) = token(64) + 10-frame history, oldest first, feature-major:
#   base ang vel (10x3), q - default (10x29), dq (10x29), last decoder action (10x29),
#   gravity direction in base frame (10x3)
HISTORY_LEN = 10
DECODER_OBS_DIM = TOKEN_DIM + HISTORY_LEN * (3 + 29 + 29 + 29 + 3)
assert DECODER_OBS_DIM == 994

CONTROL_DT = 0.02  # SONIC runs at 50 Hz

# Racket mount on right_wrist_yaw_link (table-tennis-motion-kit / VICAR model):
# racket origin at +0.2365 m along the wrist x axis, identity rotation;
# shaft = local +X, blade plane = XZ, face normal = local +Y;
# contact point = racket origin + 6.2 mm along local +Y (mesh estimate, not calibrated).
RACKET_PARENT_LINK = "right_wrist_yaw_link"
RACKET_OFFSET = (0.2365, 0.0, 0.0)
RACKET_CONTACT_OFFSET = (0.2365, 0.0062, 0.0)
RACKET_FACE_AXIS = (0.0, 1.0, 0.0)
RACKET_FACE_RADIUS = 0.075
