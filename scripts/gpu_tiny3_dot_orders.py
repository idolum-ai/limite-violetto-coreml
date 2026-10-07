"""Identify pinned CPU q·k reduction order on tiny3 layer-32 operands."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from portable.model import load_model

torch.set_num_threads(4)
OUT = ROOT / "evidence/04d-gpu/tiny3-dot-orders"
OUT.mkdir(parents=True, exist_ok=False)
ids = [87555, 130664, 114854]
model = load_model("cpu")
attn = model.backbone.layers[32].self_attn
captured = {}
old = attn.project
def project(hidden, scale, ve, cos, sin):
    result = old(hidden, scale, ve, cos, sin)
    captured["q"] = result[0].detach().clone()
    captured["k"] = result[1].detach().clone()
    return result
attn.project = project
try:
    with torch.inference_mode():
        model.reference_hidden(torch.tensor([ids], dtype=torch.long))
finally:
    attn.project = old
q_t = captured["q"].float()
k_t = captured["k"].repeat_interleave(5, dim=2).float()
native = torch.einsum("bthd,bshd->bhts", q_t, k_t).numpy()
q = q_t.numpy()[0].transpose(1, 0, 2)
k = k_t.numpy()[0].transpose(1, 0, 2)
products = np.float32(q[:, :, None, :] * k[:, None, :, :]).reshape(-1, 128)
reference = native.reshape(-1)

def metric(result):
    diff = np.abs(result.astype(np.float64) - reference.astype(np.float64))
    ix = np.flatnonzero(diff != 0)
    return {"different_elements": int(len(ix)), "max_abs": float(diff.max()),
            "first_differences": [{"index": int(i), "actual": float(result[i]),
                                   "expected": float(reference[i])} for i in ix[:8]]}

def fold_linear(acc):
    total = acc[:, 0].copy()
    for j in range(1, acc.shape[1]): total = np.float32(total + acc[:, j])
    return total

def fold_reverse(acc):
    total = acc[:, -1].copy()
    for j in range(acc.shape[1] - 2, -1, -1): total = np.float32(total + acc[:, j])
    return total

def fold_tree(acc):
    values = acc.copy()
    while values.shape[1] > 1:
        values = np.float32(values[:, 0::2] + values[:, 1::2])
    return values[:, 0]

def fold_shuffle(acc):
    # Pinned NEON vec_reduce_all: swap halves, then swap adjacent lanes.
    values = acc.copy()
    length = values.shape[1]
    stride = length // 2
    while stride >= 1:
        permutation = np.arange(length) ^ stride
        values = np.float32(values + values[:, permutation])
        stride //= 2
    return values[:, 0]

report = {"ids": ids, "shape": list(native.shape), "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
    ROOT / "scripts/gpu_tiny3_dot_orders.py", ROOT / "portable/model.py",
    ROOT / "sources/value-model/modeling_backbone.py")},
    "product_exactness_check": {"all_bf16_operands": True,
        "max_abs_product": float(np.abs(products).max())}, "experiments": []}
for lanes in (1, 2, 4, 8, 16, 32, 64, 128):
    acc = np.zeros((len(reference), lanes), dtype=np.float32)
    for j in range(128):
        lane = j % lanes
        acc[:, lane] = np.float32(acc[:, lane] + products[:, j])
    for name, fold in (("linear", fold_linear), ("reverse", fold_reverse),
                       ("tree", fold_tree), ("shuffle", fold_shuffle)):
        result = fold(acc)
        row = {"lanes": lanes, "fold": name, **metric(result)}
        report["experiments"].append(row)
        print(lanes, name, row["different_elements"], row["max_abs"], flush=True)
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
