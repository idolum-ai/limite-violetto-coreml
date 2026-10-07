"""Find first CPU rounding divergence in the all-site ordered-norm shard."""
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

import coreml_port.graphs as graphs
from coreml_port.cleanup_guard import install
from coreml_port.ordered_attention import softmax
from coreml_port.ordered_linear import linear
from coreml_port.ordered_norm import raw_norm
from portable.model import load_model

install()
torch.set_num_threads(4)
graphs.linear = linear
graphs.softmax = softmax
original_norm = graphs.norm
original_q = graphs.Q
OUT = ROOT / "evidence/04d-gpu/ordered-norm-shard"
ART = ROOT / "artifacts/04d-gpu/ordered-norm-shard"
PATH = ART / "rounding-sites.mlpackage"
if PATH.exists():
    raise RuntimeError(f"Refusing to reuse stale package: {PATH}")


class Diagnostic(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.group = graphs.Group(model, 0).eval()

    def forward(self, x, ve, positions, tap0, tap12, tap23, divisor1280):
        captured = []
        def tap_q(value):
            result = original_q(value)
            captured.append(result)
            return result
        def selected_norm(value):
            width = value.shape[-1]
            divisor = divisor1280 * (0.1 if width == 128 else 1.0)
            return graphs.Q(raw_norm(value, divisor))
        graphs.Q = tap_q
        graphs.norm = selected_norm
        try:
            y, _ = self.group(x, ve, positions, tap0, tap12, tap23)
        finally:
            graphs.Q = original_q
            graphs.norm = original_norm
        return (y, *captured)


model = load_model()
ids = torch.tensor([json.loads((ROOT / "evidence/01-reference/local-input-validation.json").read_text())["cases"][0]["input_ids"][:32]])
with torch.inference_mode():
    x, ve = graphs.Embedding(model)(ids)
graph = Diagnostic(model).eval()
args = (x, ve, torch.arange(32, dtype=torch.int32), x.clone(), x.clone(), x.clone(), torch.tensor([1280.]))
with torch.inference_mode():
    expected = graph(*args)
for name, value in graph.named_buffers():
    if "_cache_" in name:
        value.zero_()
trace = torch.jit.trace(graph, args, check_trace=False)
names = ["y"] + [f"round_{i}" for i in range(len(expected) - 1)]
input_names = ("x", "ve", "positions", "tap0", "tap12", "tap23", "divisor1280")
inputs = [ct.TensorType(name=name, shape=value.shape, dtype=np.int32 if name == "positions" else np.float32)
          for name, value in zip(input_names, args)]
states = [ct.StateType(name=name, wrapped_type=ct.TensorType(shape=value.shape, dtype=np.float16))
          for name, value in graph.named_buffers() if "_cache_" in name]
converted = ct.convert(
    trace, inputs=inputs, outputs=[ct.TensorType(name=name, dtype=np.float32) for name in names],
    states=states, compute_precision=ct.precision.FLOAT32,
    minimum_deployment_target=ct.target.macOS15, skip_model_load=True,
)
converted.save(str(PATH))
report = {
    "sources": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
        ROOT / "scripts/gpu_ordered_norm_stages.py", ROOT / "coreml_port/graphs.py",
        ROOT / "coreml_port/ordered_linear.py", ROOT / "coreml_port/ordered_attention.py",
        ROOT / "coreml_port/ordered_norm.py")},
    "input_sha256": hashlib.sha256(x.numpy().tobytes()).hexdigest(),
    "rounding_site_count": len(expected) - 1,
    "results": {},
}
for unit in (ct.ComputeUnit.CPU_ONLY, ct.ComputeUnit.CPU_AND_GPU):
    cm = ct.models.MLModel(str(PATH), compute_units=unit)
    got = cm.predict({name: value.numpy() for name, value in zip(input_names, args)}, state=cm.make_state())
    keys = [output.name for output in cm.get_spec().description.output]
    rows = []
    for name, key, want in zip(names, keys, expected):
        actual = got[key]
        diff = np.abs(actual.astype(np.float64) - want.numpy().astype(np.float64))
        ix = np.argwhere(diff != 0)
        rows.append({"name": name, "output_key": key, "shape": list(want.shape),
                     "different_elements": len(ix), "max_abs": float(diff.max()),
                     "mismatches": [{"index": row.tolist(), "got": float(actual[tuple(row)]),
                                     "want": float(want.numpy()[tuple(row)])} for row in ix[:5]]})
    report["results"][unit.name] = rows
    print(unit.name, "first", [(r["name"], r["different_elements"], r["max_abs"]) for r in rows if r["different_elements"]][:8], flush=True)
    (OUT / "rounding-sites.json").write_text(json.dumps(report, indent=2) + "\n")
