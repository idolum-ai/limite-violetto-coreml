"""Try explicit CPU vector reduction orders on the captured norm input."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

torch.set_num_threads(4)
x = torch.from_numpy(np.load(ROOT / "artifacts/04d-gpu/cpu-norm/norm-input.npz")["value"])
square = x * x
want = square.mean(-1, keepdim=True)


def reduced(vec_size, combine="balanced", horizontal="linear"):
    width = 4 * vec_size
    acc = [square[..., j * vec_size:(j + 1) * vec_size].clone() for j in range(4)]
    for i in range(1, square.shape[-1] // width):
        for j in range(4):
            acc[j] = acc[j] + square[..., i * width + j * vec_size:i * width + (j + 1) * vec_size]
    if combine == "balanced":
        lanes = (acc[0] + acc[1]) + (acc[2] + acc[3])
    else:
        lanes = ((acc[0] + acc[1]) + acc[2]) + acc[3]
    if horizontal == "linear":
        total = lanes[..., :1]
        for j in range(1, vec_size):
            total = total + lanes[..., j:j + 1]
    else:
        while lanes.shape[-1] > 1:
            half = lanes.shape[-1] // 2
            lanes = lanes[..., :half] + lanes[..., half:]
        total = lanes
    return total * (1 / square.shape[-1])


def record(value):
    diff = torch.abs(value - want)
    return {"different_rows": int(torch.count_nonzero(diff)), "max_abs": float(diff.max()),
            "row9": float(value[0, 9, 0]), "row9_want": float(want[0, 9, 0])}


report = {
    "input_sha256": __import__("hashlib").sha256(x.numpy().tobytes()).hexdigest(),
    "torch_sum_times_factor": record(square.sum(-1, keepdim=True) * (1 / 1280)),
    "torch_sum_divisor": record(square.sum(-1, keepdim=True) / 1280),
}
for vec_size in (4, 8):
    for combine in ("balanced", "linear"):
        for horizontal in ("linear", "balanced"):
            name = f"vec{vec_size}-{combine}-{horizontal}"
            report[name] = record(reduced(vec_size, combine, horizontal))
print(json.dumps(report, indent=2))
(ROOT / "evidence/04d-gpu/cpu-norm/reduction-orders.json").write_text(json.dumps(report, indent=2) + "\n")
