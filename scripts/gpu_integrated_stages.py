"""Locate first BF16 rounding difference for candidate-2 prompt 2 shard 0."""
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
from coreml_port.ordered_norm import NormWrapped
from coreml_port.ordered_xsa import ordered_xsa_dot, ordered_xsa_normalize
from portable.model import load_model

install()
torch.set_num_threads(4)
graphs.linear = linear
graphs.softmax = softmax
graphs.xsa_normalize = ordered_xsa_normalize
graphs.xsa_dot = ordered_xsa_dot
original_q = graphs.Q
OUT = ROOT / "evidence/04d-gpu/integrated-shard"
ART = ROOT / "artifacts/04d-gpu/integrated-shard"
ART.mkdir(parents=True, exist_ok=True)
PATH = ART / "prompt2-rounding-sites.mlpackage"
if PATH.exists():
    raise RuntimeError(f"Refusing to reuse stale package: {PATH}")


class Diagnostic(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.group = graphs.Group(model, 0).eval()

    def forward(self, x, ve, positions, tap0, tap12, tap23):
        captured = []
        def tap_q(value):
            result = original_q(value)
            captured.append(result)
            return result
        graphs.Q = tap_q
        try:
            y, _ = self.group(x, ve, positions, tap0, tap12, tap23)
        finally:
            graphs.Q = original_q
        return (y, *captured)


model = load_model()
case = json.loads((ROOT / "evidence/01-reference/local-input-validation.json").read_text())["cases"][2]
ids = torch.tensor([case["input_ids"][:32]], dtype=torch.int32)
divisor = torch.tensor([1280.])
with torch.inference_mode():
    x, ve = NormWrapped(graphs.Embedding(model).eval())(ids, divisor)
graph = NormWrapped(Diagnostic(model).eval()).eval()
args = (x, ve, torch.arange(32, dtype=torch.int32), x.clone(), x.clone(), x.clone(), divisor)
with torch.inference_mode():
    expected = graph(*args)
for name, value in graph.named_buffers():
    if "_cache_" in name:
        value.zero_()
trace = torch.jit.trace(graph, args, check_trace=False)
names = ["y"] + [f"round_{i}" for i in range(len(expected) - 1)]
input_names = ("x", "ve", "positions", "tap0", "tap12", "tap23", "norm_divisor")
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
        ROOT / "scripts/gpu_integrated_stages.py", ROOT / "coreml_port/graphs.py",
        ROOT / "coreml_port/ordered_linear.py", ROOT / "coreml_port/ordered_attention.py",
        ROOT / "coreml_port/ordered_norm.py", ROOT / "coreml_port/ordered_xsa.py")},
    "prompt_index": 2,
    "input_sha256": hashlib.sha256(x.numpy().tobytes()).hexdigest(),
    "rounding_site_count": len(expected) - 1,
    "results": {},
}
cm = ct.models.MLModel(str(PATH), compute_units=ct.ComputeUnit.CPU_ONLY)
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
                                 "want": float(want.numpy()[tuple(row)])} for row in ix[:10]]})
report["results"]["CPU_ONLY"] = rows
print("first", [(r["name"], r["different_elements"], r["max_abs"]) for r in rows if r["different_elements"]][:12], flush=True)
(OUT / "prompt2-rounding-sites.json").write_text(json.dumps(report, indent=2) + "\n")
