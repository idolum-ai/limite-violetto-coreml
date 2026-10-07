"""Combine dynamic reduction divisor and source-matching sqrt/div norm."""
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
x = torch.from_numpy(np.load(ROOT / "artifacts/04d-gpu/cpu-norm/norm-input.npz")["value"])
divisor = torch.full((1, 32, 1), 1280, dtype=torch.float32)


class Combined(nn.Module):
    def forward(self, x, divisor):
        mean = (x * x).sum(-1, keepdim=True) / divisor
        inv = torch.ones_like(mean) / torch.sqrt(mean + EPS)
        raw = x * inv
        return mean, inv, raw, Q(raw)


names = ("mean", "inv", "raw", "rounded")
graph = Combined().eval()
with torch.inference_mode():
    mean = (x * x).mean(-1, keepdim=True)
    inv = torch.rsqrt(mean + EPS)
    reference = (mean, inv, x * inv, Q(x * inv))
path = ROOT / "artifacts/04d-gpu/cpu-norm/combined.mlpackage"
if path.exists():
    raise RuntimeError(f"Refusing to reuse stale package: {path}")
trace = torch.jit.trace(graph, (x, divisor), check_trace=False)
converted = ct.convert(
    trace, inputs=[ct.TensorType(name="x", shape=x.shape, dtype=np.float32),
                   ct.TensorType(name="divisor", shape=divisor.shape, dtype=np.float32)],
    outputs=[ct.TensorType(name=name, dtype=np.float32) for name in names],
    compute_precision=ct.precision.FLOAT32, minimum_deployment_target=ct.target.macOS15,
    skip_model_load=True,
)
converted.save(str(path))
report = {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "input_sha256": hashlib.sha256(x.numpy().tobytes()).hexdigest(), "results": {}}
for unit in (ct.ComputeUnit.CPU_ONLY, ct.ComputeUnit.CPU_AND_GPU):
    cm = ct.models.MLModel(str(path), compute_units=unit)
    got = cm.predict({"x": x.numpy(), "divisor": divisor.numpy()})
    report["results"][unit.name] = {}
    for name, expected in zip(names, reference):
        actual = got[name]
        diff = np.abs(actual.astype(np.float64) - expected.numpy().astype(np.float64))
        ix = np.argwhere(diff != 0)
        report["results"][unit.name][name] = {
            "different_elements": len(ix), "max_abs": float(diff.max()),
            "mismatches": [{"index": row.tolist(), "got": float(actual[tuple(row)]),
                            "want": float(expected.numpy()[tuple(row)])} for row in ix[:20]],
        }
    print(unit.name, {name: (r["different_elements"], r["max_abs"]) for name, r in report["results"][unit.name].items()}, flush=True)
    (ROOT / "evidence/04d-gpu/cpu-norm/combined.json").write_text(json.dumps(report, indent=2) + "\n")
