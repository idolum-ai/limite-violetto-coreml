"""Capture exact layer-2 XSA operands from the ordered-norm Torch shard."""
import hashlib
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

import coreml_port.graphs as graphs
from coreml_port.ordered_attention import softmax
from coreml_port.ordered_linear import linear
from coreml_port.ordered_norm import ordered_sum, raw_norm
from portable.model import load_model

torch.set_num_threads(4)
graphs.linear = linear
graphs.softmax = softmax
original_norm = graphs.norm
original_q = graphs.Q
OUT = ROOT / "evidence/04d-gpu/xsa"
ART = ROOT / "artifacts/04d-gpu/xsa"
OUT.mkdir(parents=True, exist_ok=True)
ART.mkdir(parents=True, exist_ok=True)
model = load_model()
ids = torch.tensor([json.loads((ROOT / "evidence/01-reference/local-input-validation.json").read_text())["cases"][0]["input_ids"][:32]])
with torch.inference_mode():
    x, ve = graphs.Embedding(model)(ids)
group = graphs.Group(model, 0).eval()
layer = group.layers[2]
original_finish = layer.finish
captured = {}


def layer2_finish(self, x, residual, q, keys, values, v, positions, slots):
    captured["v"] = v.detach().clone()
    captured["alpha"] = self.alpha.detach().clone()
    return original_finish(x, residual, q, keys, values, v, positions, slots)


layer.finish = types.MethodType(layer2_finish, layer)
q_values = []
def tap_q(value):
    result = original_q(value)
    q_values.append(result.detach().clone())
    return result

def selected_norm(value):
    width = value.shape[-1]
    divisor = torch.tensor([1280.]) * (0.1 if width == 128 else 1.0)
    return graphs.Q(raw_norm(value, divisor))

graphs.Q = tap_q
graphs.norm = selected_norm
try:
    with torch.inference_mode():
        group(x, ve, torch.arange(32, dtype=torch.int32), x.clone(), x.clone(), x.clone())
finally:
    graphs.Q = original_q
    graphs.norm = original_norm
assert len(q_values) == 127
y = q_values[81]
v = captured["v"]
alpha = captured["alpha"]
vrep = v.repeat_interleave(5, dim=2)
with torch.inference_mode():
    vn = F.normalize(vrep, dim=-1, eps=1e-4)
    product = y * vn
    dot = product.sum(-1, keepdim=True)
    correction = alpha * dot * vn
    rounded = original_q(correction)
    ordered_vn = vrep * (torch.ones_like(ordered_sum(vrep * vrep)) / torch.sqrt(ordered_sum(vrep * vrep)))
    ordered_dot = ordered_sum(product)
    ordered_rounded = original_q(alpha * ordered_dot * vn)
    ordered_both = original_q(alpha * ordered_sum(y * ordered_vn) * ordered_vn)


def metric(a, b):
    delta = torch.abs(a.float() - b.float())
    return {"different_elements": int(torch.count_nonzero(a != b)), "max_abs": float(delta.max())}


arrays = {"y": y.numpy(), "v": v.numpy(), "alpha": alpha.numpy(),
          "vn": vn.numpy(), "product": product.numpy(), "dot": dot.numpy(),
          "correction": correction.numpy(), "rounded": rounded.numpy()}
np.savez_compressed(ART / "layer2-inputs.npz", **arrays)
report = {
    "sources": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
        ROOT / "scripts/gpu_xsa_capture.py", ROOT / "coreml_port/graphs.py",
        ROOT / "coreml_port/ordered_linear.py", ROOT / "coreml_port/ordered_attention.py",
        ROOT / "coreml_port/ordered_norm.py")},
    "input_hashes": {name: hashlib.sha256(array.tobytes()).hexdigest() for name, array in arrays.items()},
    "shapes": {name: list(array.shape) for name, array in arrays.items()},
    "torch_ordered_vn_vs_native": metric(ordered_vn, vn),
    "torch_ordered_dot_vs_native": metric(ordered_dot, dot),
    "torch_ordered_dot_round_vs_native": metric(ordered_rounded, rounded),
    "torch_ordered_both_round_vs_native": metric(ordered_both, rounded),
    "round82_reference_values": [{"index": [0, 3, 1, i], "value": float(rounded[0, 3, 1, i]),
                                   "raw": float(correction[0, 3, 1, i])} for i in (8, 81)],
}
(OUT / "capture.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
