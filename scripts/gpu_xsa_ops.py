"""Expose layer-2 XSA normalization and correction arithmetic in Core ML."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from coreml_port.cleanup_guard import install
from coreml_port.graphs import Q
from coreml_port.ordered_norm import ordered_sum

install()
torch.set_num_threads(4)
OUT = ROOT / "evidence/04d-gpu/xsa"
ART = ROOT / "artifacts/04d-gpu/xsa"
data = np.load(ART / "layer2-inputs.npz")
y = torch.from_numpy(data["y"])
v = torch.from_numpy(data["v"])
alpha = torch.from_numpy(data["alpha"])


class Operators(nn.Module):
    def forward(self, y, v, alpha):
        vrep = v.repeat_interleave(5, dim=2)
        norm = vrep.norm(p=2, dim=-1, keepdim=True)
        vn = F.normalize(vrep, dim=-1, eps=1e-4)
        product = y * vn
        dot = product.sum(-1, keepdim=True)
        scaled = alpha * dot
        correction = scaled * vn
        return norm, vn, product, dot, scaled, correction, Q(correction)


names = ("norm", "vn", "product", "dot", "scaled", "correction", "rounded")
graph = Operators().eval()
with torch.inference_mode():
    expected = graph(y, v, alpha)
    vrep = v.repeat_interleave(5, dim=2)
    ordered_vn = vrep * (torch.ones_like(ordered_sum(vrep * vrep)) / torch.sqrt(ordered_sum(vrep * vrep)))
path = ART / "operators.mlpackage"
if path.exists():
    raise RuntimeError(f"Refusing to reuse stale package: {path}")
trace = torch.jit.trace(graph, (y, v, alpha), check_trace=False)
converted = ct.convert(
    trace, inputs=[ct.TensorType(name="y", shape=y.shape, dtype=np.float32),
                   ct.TensorType(name="v", shape=v.shape, dtype=np.float32),
                   ct.TensorType(name="alpha", shape=alpha.shape, dtype=np.float32)],
    outputs=[ct.TensorType(name=name, dtype=np.float32) for name in names],
    compute_precision=ct.precision.FLOAT32, minimum_deployment_target=ct.target.macOS15,
    skip_model_load=True,
)
converted.save(str(path))
report = {
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "input_hashes": {name: hashlib.sha256(data[name].tobytes()).hexdigest() for name in ("y", "v", "alpha")},
    "results": {},
}
for unit in (ct.ComputeUnit.CPU_ONLY, ct.ComputeUnit.CPU_AND_GPU):
    cm = ct.models.MLModel(str(path), compute_units=unit)
    got = cm.predict({"y": y.numpy(), "v": v.numpy(), "alpha": alpha.numpy()})
    report["results"][unit.name] = {}
    for name, want in zip(names, expected):
        actual = got[name]
        diff = np.abs(actual.astype(np.float64) - want.numpy().astype(np.float64))
        ix = np.argwhere(diff != 0)
        report["results"][unit.name][name] = {
            "different_elements": len(ix), "max_abs": float(diff.max()),
            "mismatches": [{"index": row.tolist(), "got": float(actual[tuple(row)]),
                            "want": float(want.numpy()[tuple(row)])} for row in ix[:20]],
        }
    ordered_diff = np.abs(got["vn"].astype(np.float64) - ordered_vn.numpy().astype(np.float64))
    report["results"][unit.name]["vn_vs_ordered_candidate"] = {
        "different_elements": int(np.count_nonzero(ordered_diff)), "max_abs": float(ordered_diff.max())}
    print(unit.name, {name: (r["different_elements"], r["max_abs"]) for name, r in report["results"][unit.name].items()}, flush=True)
    (OUT / "operators.json").write_text(json.dumps(report, indent=2) + "\n")
