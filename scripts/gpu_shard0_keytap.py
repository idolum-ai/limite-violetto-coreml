"""Expose the first key raw/rounded site in the Q_even four-layer shard."""
import hashlib
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import coremltools as ct
import numpy as np
import torch
from torch import nn

import coreml_port.graphs as graphs
from coreml_port.cleanup_guard import install
from coreml_port.ordered_linear import linear
from portable.model import load_model
from scripts.gpu_even_round import project_layer0_key_even

install()
torch.set_num_threads(4)
graphs.linear = linear


class Diagnostic(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.group = graphs.Group(model, 0).eval()
        self.group.layers[0].project = types.MethodType(self.key_project, self.group.layers[0])
        self.capture = None

    def key_project(self, layer, x, ve, cos, sin):
        return project_layer0_key_even(layer, x, ve, cos, sin, self.capture)

    def forward(self, x, ve, positions, tap0, tap12, tap23):
        held = []
        self.capture = lambda raw, rounded: held.extend((raw, rounded))
        try:
            y, _ = self.group(x, ve, positions, tap0, tap12, tap23)
        finally:
            self.capture = None
        return y, held[0], held[1]


model = load_model()
ids = torch.tensor([json.loads((ROOT / "evidence/01-reference/local-input-validation.json").read_text())["cases"][0]["input_ids"][:32]])
with torch.inference_mode():
    x, ve = graphs.Embedding(model)(ids)
graph = Diagnostic(model).eval()
args = (x, ve, torch.arange(32, dtype=torch.int32), x.clone(), x.clone(), x.clone())
with torch.inference_mode():
    want = graph(*args)
for name, value in graph.named_buffers():
    if "_cache_" in name:
        value.zero_()
trace = torch.jit.trace(graph, args, check_trace=False)
names = ("y", "kraw", "k")
inputs = [ct.TensorType(name=name, shape=value.shape, dtype=np.int32 if name == "positions" else np.float32)
          for name, value in zip(("x", "ve", "positions", "tap0", "tap12", "tap23"), args)]
states = [ct.StateType(name=name, wrapped_type=ct.TensorType(shape=value.shape, dtype=np.float16))
          for name, value in graph.named_buffers() if "_cache_" in name]
path = ROOT / "artifacts/04d-gpu/shard0-keytap.mlpackage"
if path.exists():
    raise RuntimeError(f"Refusing to reuse stale package: {path}")
converted = ct.convert(
    trace, inputs=inputs, outputs=[ct.TensorType(name=name, dtype=np.float32) for name in names],
    states=states, compute_precision=ct.precision.FLOAT32,
    minimum_deployment_target=ct.target.macOS15, skip_model_load=True,
)
converted.save(str(path))
report = {
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "round_module_sha256": hashlib.sha256((ROOT / "scripts/gpu_even_round.py").read_bytes()).hexdigest(),
    "input_sha256": hashlib.sha256(x.numpy().tobytes()).hexdigest(),
    "results": {},
}
for unit in (ct.ComputeUnit.CPU_ONLY, ct.ComputeUnit.CPU_AND_GPU):
    cm = ct.models.MLModel(str(path), compute_units=unit)
    got = cm.predict({name: value.numpy() for name, value in zip(("x", "ve", "positions", "tap0", "tap12", "tap23"), args)}, state=cm.make_state())
    report["results"][unit.name] = {}
    for name, expected in zip(names, want):
        actual = got[name]
        diff = np.abs(actual.astype(np.float64) - expected.numpy().astype(np.float64))
        ix = np.argwhere(diff != 0)
        report["results"][unit.name][name] = {
            "different_elements": len(ix), "max_abs": float(diff.max()),
            "mismatches": [{"index": row.tolist(), "got": float(actual[tuple(row)]),
                            "want": float(expected.numpy()[tuple(row)])} for row in ix[:10]],
        }
    print(unit.name, {n: (r["different_elements"], r["max_abs"]) for n, r in report["results"][unit.name].items()}, flush=True)
out = ROOT / "evidence/04d-gpu/keytap.json"
out.write_text(json.dumps(report, indent=2) + "\n")
