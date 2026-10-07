"""Test explicit round-to-even for composed layer-0 key projection."""
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
from coreml_port.graphs import Q, Embedding, Layer
from coreml_port.ordered_linear import linear
from portable.model import load_model
from scripts.gpu_even_round import Q_even

install()
torch.set_num_threads(4)


class QKVEven(nn.Module):
    def __init__(self, layer):
        super().__init__()
        for name in ("q", "k", "v"):
            self.register_buffer(name, getattr(layer, name))

    def forward(self, x):
        q = Q(linear(x, self.q))
        kraw = linear(x, self.k)
        k = Q_even(kraw)
        v = Q(linear(x, self.v))
        return q, kraw, k, v


model = load_model()
ids = torch.tensor([json.loads((ROOT / "evidence/01-reference/local-input-validation.json").read_text())["cases"][0]["input_ids"][:32]])
with torch.inference_mode():
    x, _ = Embedding(model)(ids)
    layer = Layer(model, 0)
graph = QKVEven(layer).eval()
with torch.inference_mode():
    want = graph(x)
    baseline = Q(want[1])
assert torch.equal(want[2], baseline), "Even implementation differs from Torch Q on this input"

path = ROOT / "artifacts/04d-gpu/qkv-even.mlpackage"
if path.exists():
    raise RuntimeError(f"Refusing to reuse stale package: {path}")
trace = torch.jit.trace(graph, x[:, :3], check_trace=False)
names = ("q", "kraw", "k", "v")
converted = ct.convert(
    trace,
    inputs=[ct.TensorType(name="x", shape=(1, ct.RangeDim(1, 32, default=3), 1280), dtype=np.float32)],
    outputs=[ct.TensorType(name=n, dtype=np.float32) for n in names],
    compute_precision=ct.precision.FLOAT32,
    minimum_deployment_target=ct.target.macOS15,
    skip_model_load=True,
)
converted.save(str(path))
report = {
    "input_sha256": hashlib.sha256(x.numpy().tobytes()).hexdigest(),
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "torch_even_equals_original": True,
    "runs": {},
}
for unit in (ct.ComputeUnit.CPU_ONLY, ct.ComputeUnit.CPU_AND_GPU):
    cm = ct.models.MLModel(str(path), compute_units=unit)
    got = cm.predict({"x": x.numpy()})
    report["runs"][unit.name] = {}
    for name, expected in zip(names, want):
        actual = got[name]
        diff = np.abs(actual.astype(np.float64) - expected.numpy().astype(np.float64))
        ix = np.argwhere(diff != 0)
        report["runs"][unit.name][name] = {
            "different_elements": len(ix), "max_abs": float(diff.max()),
            "mismatches": [{"index": row.tolist(), "got": float(actual[tuple(row)]),
                            "want": float(expected.numpy()[tuple(row)])} for row in ix[:20]],
        }
    print(unit.name, {n: report["runs"][unit.name][n]["different_elements"] for n in names}, flush=True)
(ROOT / "evidence/04d-gpu/key-even.json").write_text(json.dumps(report, indent=2) + "\n")
