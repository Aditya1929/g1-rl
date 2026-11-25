"""ZMQ token publisher for gear_sonic_deploy (protocol v4), same wire format as
sonic-dataset/stream_tokens.py and gear_sonic/scripts/run_vla_inference.py:

    [topic "pose"][1280-byte JSON header][float32 token (1, 64)][int64 frame_index]

The deploy controller must run with --input-type zmq and have ZMQ streaming enabled.
"""
from __future__ import annotations

import json

import numpy as np

HEADER_SIZE = 1280


def pack(token: np.ndarray, frame_index: int, topic: str = "pose") -> bytes:
    token = np.ascontiguousarray(token, dtype="<f4").reshape(1, -1)
    assert token.shape[1] == 64, token.shape
    frame = np.array([frame_index], dtype="<i8")
    header = json.dumps({"v": 4, "endian": "le", "count": 1, "fields": [
        {"name": "token_state", "dtype": "f32", "shape": list(token.shape)},
        {"name": "frame_index", "dtype": "i64", "shape": [1]},
    ]}, separators=(",", ":")).encode()
    assert len(header) <= HEADER_SIZE
    return topic.encode() + header.ljust(HEADER_SIZE, b"\x00") + token.tobytes() + frame.tobytes()


class TokenPublisher:
    def __init__(self, port: int = 5556, live: bool = False):
        self.live, self.frame = live, 0
        if live:
            import zmq

            self.sock = zmq.Context.instance().socket(zmq.PUB)
            self.sock.bind(f"tcp://*:{port}")

    def send(self, token: np.ndarray):
        if self.live:
            self.sock.send(pack(token, self.frame))
        self.frame += 1
