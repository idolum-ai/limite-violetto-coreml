"""Isolate layer-2 RMS normalization on the exact ordered-softmax input."""
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
from portable.model import EPS, load_model

install()
torch.set_num_threads(4)
graphs.linear = linear
graphs.softmax = softmax
OUT = ROOT / "evidence/04d-gpu/cpu-norm"
ART = ROOT / "artifacts/04d-gpu/cpu-norm"
OUT.mkdir(parents=True, exist_ok=True)
ART.mkdir(parents=True, exist_ok=True)
ids = torch.tensor([json.loads((ROOT / "evidence/01-reference/local-input-validation.json").read_text())["cases"][0]["input_ids"][:32]])
model = load_model()
with torch.inference_mode():
    x, ve = graphs.Embedding(model)(ids)
group = graphs.Group(model, 0).eval()
args = (x, ve, torch.arange(32, dtype=torch.int32), x.clone(), x.clone(), x.clone())
captured = []
original_q = graphs.Q


def tap_q(value):
    result = original_q(value)
    captured.append(result)
    return result


graphs.Q = tap_q
try:
    with torch.inference_mode():
        group(*args)
finally:
    graphs.Q = original_q
assert len(captured) > 69
norm_input = captured[68].detach().clone()
assert norm_input.shape == (1, 32, 1280)


class NormStages(nn.Module):
    def forward(self, value):
        square = value * value
        mean = square.mean(-1, keepdim=True)
        mean_eps = mean + EPS
        inv = torch.rsqrt(mean_eps)
        raw = value * inv
        rounded = graphs.Q(raw)
        return square, mean, mean_eps, inv, raw, rounded


names = ("square", "mean", "mean_eps", "inv", "raw", "rounded")
graph = NormStages().eval()
with torch.inference_mode():
    want = graph(norm_input)
stage_model = ct.models.MLModel(str(ROOT / "artifacts/04d-cpu/ordered-softmax-stages/rounding-sites.mlpackage"), compute_units=ct.ComputeUnit.CPU_ONLY)
stage_input = {name: value.numpy() for name, value in zip(("x", "ve", "positions", "tap0", "tap12", "tap23"), args)}
stage_output = stage_model.predict(stage_input, state=stage_model.make_state())
stage_value = stage_output["round_68"]

def comparison(actual, expected):
    diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    index = np.argwhere(diff != 0)
    return {
        "different_elements": len(index), "max_abs": float(diff.max()),
        "mismatches": [{"index": row.tolist(), "got": float(actual[tuple(row)]),
                        "want": float(expected[tuple(row)])} for row in index[:20]],
    }


package = ART / "norm-stages.mlpackage"
if package.exists():
    raise RuntimeError(f"Refusing to reuse stale package: {package}")
trace = torch.jit.trace(graph, norm_input, check_trace=False)
converted = ct.convert(
    trace, inputs=[ct.TensorType(name="value", shape=norm_input.shape, dtype=np.float32)],
    outputs=[ct.TensorType(name=name, dtype=np.float32) for name in names],
    compute_precision=ct.precision.FLOAT32, minimum_deployment_target=ct.target.macOS15,
    skip_model_load=True,
)
converted.save(str(package))
report = {
    "sources": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
        ROOT / "scripts/gpu_cpu_norm.py", ROOT / "coreml_port/graphs.py",
        ROOT / "coreml_port/ordered_linear.py", ROOT / "coreml_port/ordered_attention.py")},
    "norm_input_sha256": hashlib.sha256(norm_input.numpy().tobytes()).hexdigest(),
    "norm_input_shape": list(norm_input.shape),
    "stage_round68_vs_torch": comparison(stage_value, norm_input.numpy()),
    "results": {},
}
np.savez_compressed(ART / "norm-input.npz", value=norm_input.numpy())
for unit in (ct.ComputeUnit.CPU_ONLY, ct.ComputeUnit.CPU_AND_GPU):
    cm = ct.models.MLModel(str(package), compute_units=unit)
    got = cm.predict({"value": norm_input.numpy()})
    report["results"][unit.name] = {name: comparison(got[name], expected.numpy()) for name, expected in zip(names, want)}
    np.savez_compressed(ART / f"norm-captures-{unit.name}.npz", **{f"{name}_got": got[name] for name in names}, **{f"{name}_want": expected.numpy() for name, expected in zip(names, want)})
    print(unit.name, {name: (v["different_elements"], v["max_abs"]) for name, v in report["results"][unit.name].items()}, flush=True)
    (OUT / "norm-stages.json").write_text(json.dumps(report, indent=2) + "\n")
