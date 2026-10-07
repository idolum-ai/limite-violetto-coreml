"""Test explicit Torch CPU vector norm and XSA dot order in Core ML."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import coremltools as ct
import numpy as np
import torch
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


def ordered_l2(value):
    width = value.shape[-1]
    assert width == 128
    lanes = value[..., :4] * value[..., :4]
    for j in range(4, width, 4):
        q = value[..., j:j + 4]
        lanes = lanes + q * q
    squared = ((lanes[..., 0:1] + lanes[..., 1:2]) + lanes[..., 2:3]) + lanes[..., 3:4]
    return torch.sqrt(squared)


class OrderedXSA(nn.Module):
    def forward(self, y, v, alpha):
        vrep = v.repeat_interleave(5, dim=2)
        norm = ordered_l2(vrep)
        vn = vrep / norm.clamp_min(1e-4)
        product = y * vn
        dot = ordered_sum(product)
        correction = alpha * dot * vn
        return norm, vn, product, dot, correction, Q(correction)


names = ("norm", "vn", "product", "dot", "correction", "rounded")
graph = OrderedXSA().eval()
with torch.inference_mode():
    expected = graph(y, v, alpha)
assert torch.equal(expected[1], torch.from_numpy(data["vn"]))
assert torch.equal(expected[3], torch.from_numpy(data["dot"]))
assert torch.equal(expected[5], torch.from_numpy(data["rounded"]))
path = ART / "ordered.mlpackage"
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
    "ordered_norm_sha256": hashlib.sha256((ROOT / "coreml_port/ordered_norm.py").read_bytes()).hexdigest(),
    "input_hashes": {name: hashlib.sha256(data[name].tobytes()).hexdigest() for name in ("y", "v", "alpha")},
    "torch_candidate_exact": True, "results": {},
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
    print(unit.name, {name: (r["different_elements"], r["max_abs"]) for name, r in report["results"][unit.name].items()}, flush=True)
    (OUT / "ordered.json").write_text(json.dumps(report, indent=2) + "\n")
