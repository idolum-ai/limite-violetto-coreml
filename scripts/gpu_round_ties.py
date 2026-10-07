"""Determine Core ML rounding semantics at exact BF16 midpoints."""
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

install()
torch.set_num_threads(4)


class RoundTies(nn.Module):
    def forward(self, x):
        return Q(x), torch.round(x * 256) / 256


x = np.full((1, 32, 256), 0.791015625, dtype=np.float32)
x[:, 1::2] *= -1
x[:, :, 1::4] = 0.794921875  # midpoint whose even neighbor is upper
x[:, :, 2::4] = -0.794921875
x[:, :, 3::4] = 0.7890625
graph = RoundTies().eval()
path = ROOT / "artifacts/04d-gpu/round-ties.mlpackage"
if path.exists():
    raise RuntimeError(f"Refusing to reuse stale package: {path}")
trace = torch.jit.trace(graph, torch.from_numpy(x), check_trace=False)
cm = ct.convert(
    trace, inputs=[ct.TensorType(name="x", shape=x.shape, dtype=np.float32)],
    outputs=[ct.TensorType(name="q", dtype=np.float32), ct.TensorType(name="simple", dtype=np.float32)],
    compute_precision=ct.precision.FLOAT32, minimum_deployment_target=ct.target.macOS15,
    skip_model_load=True,
)
cm.save(str(path))
with torch.inference_mode():
    want = [a.numpy() for a in graph(torch.from_numpy(x))]
results = {}
for unit in (ct.ComputeUnit.CPU_ONLY, ct.ComputeUnit.CPU_AND_GPU):
    model = ct.models.MLModel(str(path), compute_units=unit)
    got = model.predict({"x": x})
    results[unit.name] = {}
    for name, expected in zip(("q", "simple"), want):
        actual = got[name]
        index = np.argwhere(actual != expected)
        results[unit.name][name] = {
            "different_elements": len(index), "max_abs": float(np.abs(actual - expected).max()),
            "examples": [{"index": row.tolist(), "input": float(x[tuple(row)]),
                          "got": float(actual[tuple(row)]), "want": float(expected[tuple(row)])}
                         for row in index[:8]],
        }
    print(unit.name, results[unit.name], flush=True)
out = {
    "input_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "results": results,
}
(ROOT / "evidence/04d-gpu/round-ties.json").write_text(json.dumps(out, indent=2) + "\n")
