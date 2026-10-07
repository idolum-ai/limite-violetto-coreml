"""Separate Core ML layer-2 norm reduction from divisor implementation."""
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

install()
torch.set_num_threads(4)
norm_input = torch.from_numpy(np.load(ROOT / "artifacts/04d-gpu/cpu-norm/norm-input.npz")["value"])
square = norm_input * norm_input
divisor = torch.full((1, 32, 1), 1280, dtype=torch.float32)


class Sums(nn.Module):
    def forward(self, value, divisor):
        total = value.sum(-1, keepdim=True)
        return total, total / divisor, total / 1280, total * (1 / 1280), value.mean(-1, keepdim=True)


names = ("sum", "dynamic_div", "const_div", "const_mul", "mean")
graph = Sums().eval()
with torch.inference_mode():
    want = graph(square, divisor)
path = ROOT / "artifacts/04d-gpu/cpu-norm/sumdiv.mlpackage"
if path.exists():
    raise RuntimeError(f"Refusing to reuse stale package: {path}")
trace = torch.jit.trace(graph, (square, divisor), check_trace=False)
converted = ct.convert(
    trace,
    inputs=[ct.TensorType(name="value", shape=square.shape, dtype=np.float32),
            ct.TensorType(name="divisor", shape=divisor.shape, dtype=np.float32)],
    outputs=[ct.TensorType(name=name, dtype=np.float32) for name in names],
    compute_precision=ct.precision.FLOAT32,
    minimum_deployment_target=ct.target.macOS15,
    skip_model_load=True,
)
converted.save(str(path))
report = {
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "square_sha256": hashlib.sha256(square.numpy().tobytes()).hexdigest(),
    "results": {},
}
for unit in (ct.ComputeUnit.CPU_ONLY, ct.ComputeUnit.CPU_AND_GPU):
    cm = ct.models.MLModel(str(path), compute_units=unit)
    got = cm.predict({"value": square.numpy(), "divisor": divisor.numpy()})
    report["results"][unit.name] = {}
    for name, expected in zip(names, want):
        actual = got[name]
        diff = np.abs(actual.astype(np.float64) - expected.numpy().astype(np.float64))
        index = np.argwhere(diff != 0)
        report["results"][unit.name][name] = {
            "different_rows": len(index), "max_abs": float(diff.max()),
            "row9": float(actual[0, 9, 0]), "row9_want": float(expected[0, 9, 0]),
            "mismatches": [{"index": row.tolist(), "got": float(actual[tuple(row)]),
                            "want": float(expected.numpy()[tuple(row)])} for row in index[:20]],
        }
    print(unit.name, {name: (r["different_rows"], r["max_abs"]) for name, r in report["results"][unit.name].items()}, flush=True)
    (ROOT / "evidence/04d-gpu/cpu-norm/sumdiv.json").write_text(json.dumps(report, indent=2) + "\n")
