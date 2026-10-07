"""Test runtime-denominator RMS norm at the layer-2 CPU boundary."""
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
from portable.model import EPS

install()
torch.set_num_threads(4)
value = torch.from_numpy(np.load(ROOT / "artifacts/04d-gpu/cpu-norm/norm-input.npz")["value"])
divisor = torch.full((1, 32, 1), 1280, dtype=torch.float32)


class DynamicNorm(nn.Module):
    def forward(self, x, divisor):
        square = x * x
        mean = square.sum(-1, keepdim=True) / divisor
        inv = torch.rsqrt(mean + EPS)
        raw = x * inv
        return mean, inv, raw, Q(raw)


graph = DynamicNorm().eval()
names = ("mean", "inv", "raw", "rounded")
with torch.inference_mode():
    want = graph(value, divisor)
    baseline = Q(value * torch.rsqrt((value * value).mean(-1, keepdim=True) + EPS))
assert torch.equal(want[-1], baseline)
path = ROOT / "artifacts/04d-gpu/cpu-norm/dynamic-norm.mlpackage"
if path.exists():
    raise RuntimeError(f"Refusing to reuse stale package: {path}")
trace = torch.jit.trace(graph, (value, divisor), check_trace=False)
converted = ct.convert(
    trace, inputs=[ct.TensorType(name="x", shape=value.shape, dtype=np.float32),
                   ct.TensorType(name="divisor", shape=divisor.shape, dtype=np.float32)],
    outputs=[ct.TensorType(name=name, dtype=np.float32) for name in names],
    compute_precision=ct.precision.FLOAT32, minimum_deployment_target=ct.target.macOS15,
    skip_model_load=True,
)
converted.save(str(path))
report = {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "input_sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest(), "results": {}}
for unit in (ct.ComputeUnit.CPU_ONLY, ct.ComputeUnit.CPU_AND_GPU):
    cm = ct.models.MLModel(str(path), compute_units=unit)
    got = cm.predict({"x": value.numpy(), "divisor": divisor.numpy()})
    report["results"][unit.name] = {}
    for name, expected in zip(names, want):
        actual = got[name]
        diff = np.abs(actual.astype(np.float64) - expected.numpy().astype(np.float64))
        index = np.argwhere(diff != 0)
        report["results"][unit.name][name] = {
            "different_elements": len(index), "max_abs": float(diff.max()),
            "mismatches": [{"index": row.tolist(), "got": float(actual[tuple(row)]),
                            "want": float(expected.numpy()[tuple(row)])} for row in index[:20]],
        }
    print(unit.name, {name: (r["different_elements"], r["max_abs"]) for name, r in report["results"][unit.name].items()}, flush=True)
    (ROOT / "evidence/04d-gpu/cpu-norm/dynamic-norm.json").write_text(json.dumps(report, indent=2) + "\n")
