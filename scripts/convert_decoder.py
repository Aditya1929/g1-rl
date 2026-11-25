"""Extract the SONIC decoder MLP from ONNX into data/sonic_decoder_<policy>.npz.

The release decoder graph is: obs(994) -> [Linear -> SiLU] x6 -> Linear -> action(29)
(the leading Slice/Concat just re-joins token(64) and history(930) in order).
This script checks that structure, saves the weights, and (if onnxruntime is
importable) verifies a numpy re-implementation against onnxruntime.

  ~/sonic-dataset/.venv/bin/python scripts/convert_decoder.py   # has onnx + onnxruntime
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/g1_sonic_tt"))
from g1_sonic_tt import paths  # noqa: E402
from g1_sonic_tt.sonic.constants import DECODER_OBS_DIM, TOKEN_DIM  # noqa: E402


def extract(onnx_path: Path):
    g = onnx.load(str(onnx_path)).graph
    init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    consts = {n.output[0]: numpy_helper.to_array(n.attribute[0].t) for n in g.node if n.op_type == "Constant"}
    ops = [n for n in g.node if n.op_type != "Constant"]
    # input re-join: Slice[0:-930], Slice[-930:], Unsqueeze, Unsqueeze, Concat(-1)  == identity
    slices = [n for n in ops if n.op_type == "Slice" and n.input[0] == g.input[0].name]
    assert len(slices) == 2
    starts = sorted(int(consts[s.input[1]][0]) for s in slices)
    assert starts == [-(DECODER_OBS_DIM - TOKEN_DIM), 0], starts
    concat = [n for n in ops if n.op_type == "Concat"]
    assert len(concat) == 1 and concat[0].input[0].startswith("/Unsqueeze_output") and concat[0].input[1].startswith("/Unsqueeze_1"), \
        "unexpected input ordering"
    weights, biases = [], []
    for n in ops:
        if n.op_type == "MatMul":
            weights.append(init[n.input[1]])
        elif n.op_type == "Add" and n.input[0] in init:
            biases.append(init[n.input[0]])
    acts = [n.op_type for n in ops if n.op_type == "Sigmoid"]
    assert len(weights) == len(biases) == 7 and len(acts) == 6, (len(weights), len(biases), len(acts))
    assert weights[0].shape[0] == DECODER_OBS_DIM and weights[-1].shape[1] == 29
    return weights, biases


def forward_np(x, weights, biases):
    h = x
    for i, (W, b) in enumerate(zip(weights, biases)):
        h = h @ W + b
        if i < len(weights) - 1:
            h = h / (1.0 + np.exp(-h))  # SiLU
    return h


def main():
    weights, biases = extract(paths.SONIC_DECODER_ONNX)
    out = paths.DECODER_WEIGHTS
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **{f"W{i}": w for i, w in enumerate(weights)}, **{f"b{i}": b for i, b in enumerate(biases)})
    print(f"saved {out}: layers {[w.shape for w in weights]}")
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not available; skipped parity check")
        return
    sess = ort.InferenceSession(str(paths.SONIC_DECODER_ONNX), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(32):
        x = rng.normal(0, 0.5, (1, DECODER_OBS_DIM)).astype(np.float32)
        ref = sess.run(None, {sess.get_inputs()[0].name: x})[0].reshape(-1)
        mine = forward_np(x.astype(np.float64), weights, biases).reshape(-1)
        worst = max(worst, float(np.abs(ref - mine).max()))
    print(f"parity vs onnxruntime: max abs diff {worst:.2e}")
    assert worst < 1e-3


if __name__ == "__main__":
    main()
