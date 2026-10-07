"""Shard0 trial: source-ordered RMS norm and XSA arithmetic."""
import gc
import hashlib
import json
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import coremltools as ct
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import coreml_port.graphs as graphs
from coreml_port.cleanup_guard import install
from coreml_port.ordered_attention import softmax
from coreml_port.ordered_linear import linear
from coreml_port.ordered_norm import raw_norm
from portable.model import load_model
from scripts.gpu_xsa_math import ordered_xsa_dot, ordered_xsa_normalize
from scripts.probe_full_model import metrics

install()
torch.set_num_threads(4)
graphs.linear = linear
graphs.softmax = softmax
original_norm = graphs.norm
OUT = ROOT / "evidence/04d-gpu/xsa"
ART = ROOT / "artifacts/04d-gpu/xsa"
OUT.mkdir(parents=True, exist_ok=True)
ART.mkdir(parents=True, exist_ok=True)
PACKAGE = ART / "shard0.mlpackage"
if PACKAGE.exists():
    raise RuntimeError(f"Refusing to reuse stale package: {PACKAGE}")


def corrected_norm(value, divisor1280):
    width = value.shape[-1]
    assert width in (128, 1280)
    divisor = divisor1280 * (0.1 if width == 128 else 1.0)
    return graphs.Q(raw_norm(value, divisor))


def finish_ordered_xsa(self, x, residual, q, keys, values, v, positions, slots):
    count = q.shape[1]
    q = torch.cat((q, torch.zeros(1, 31, 10, 128)), dim=1)[:, :32]
    pos = torch.cat((positions, torch.zeros(31, dtype=torch.int32)))[:32]
    k = keys.repeat_interleave(5, dim=2).permute(0, 2, 3, 1)
    val = values.repeat_interleave(5, dim=2).permute(0, 2, 1, 3)
    scores = torch.matmul(q.permute(0, 2, 1, 3), k) * .1
    mask = slots[None, :] <= pos[:, None]
    if self.window >= 0:
        mask = mask & (slots[None, :] >= pos[:, None] - self.window)
    weights = graphs.softmax(scores.masked_fill(~mask[None, None], float("-inf")))
    y = graphs.Q(torch.matmul(weights, val).permute(0, 2, 1, 3)[:, :count])
    vn = ordered_xsa_normalize(v.repeat_interleave(5, dim=2))
    y = graphs.Q(y - graphs.Q(self.alpha * ordered_xsa_dot(y, vn) * vn))
    gate = graphs.Q(2 * graphs.Q(torch.sigmoid(graphs.Q(graphs.linear(x[..., :128], self.ag)))))
    y = graphs.Q(y * gate.unsqueeze(-1))
    out = graphs.Q(graphs.linear(y.reshape(1, -1, 1280), self.o))
    mixed = graphs.Q(graphs.Q(self.resid_attn * residual) + graphs.Q(self.post_attn * out))
    h = graphs.norm(mixed)
    activated = graphs.Q(graphs.Q(F.silu(graphs.Q(graphs.linear(h, self.gate)))) * graphs.Q(graphs.linear(h, self.up)))
    mlp = graphs.Q(graphs.linear(activated, self.down))
    return graphs.Q(graphs.Q(self.resid_mlp * mixed) + graphs.Q(self.post_mlp * mlp))


class NormShard(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.group = graphs.Group(model, 0).eval()
        for layer in self.group.layers:
            layer.finish = types.MethodType(finish_ordered_xsa, layer)
        self.norm_calls = None

    def forward(self, x, ve, positions, tap0, tap12, tap23, divisor1280):
        calls = []
        def selected_norm(value):
            index = len(calls)
            calls.append(tuple(value.shape))
            return corrected_norm(value, divisor1280)
        graphs.norm = selected_norm
        try:
            y, tap = self.group(x, ve, positions, tap0, tap12, tap23)
        finally:
            graphs.norm = original_norm
        self.norm_calls = calls
        assert len(calls) == 16
        return y, tap


ids = json.loads((ROOT / "evidence/01-reference/local-input-validation.json").read_text())["cases"][0]["input_ids"]
model = load_model()
embedding = graphs.Embedding(model).eval()
baseline = graphs.Group(model, 0).eval()
candidate = NormShard(model).eval()
del model
gc.collect()
report = {
    "sources": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
        ROOT / "scripts/gpu_xsa_shard.py", ROOT / "scripts/gpu_xsa_math.py", ROOT / "coreml_port/graphs.py",
        ROOT / "coreml_port/ordered_linear.py", ROOT / "coreml_port/ordered_attention.py",
        ROOT / "coreml_port/ordered_norm.py")},
    "norm_sites_replaced": "all 16 calls in shard 0",
    "torch_candidate_vs_original": [],
    "coreml_vs_original_torch": {},
}
samples = []
with torch.inference_mode():
    for start, end in ((0, 32), (32, 33), (33, 36)):
        x, ve = embedding(torch.tensor([ids[start:end]]))
        pos = torch.arange(start, end, dtype=torch.int32)
        divisor = torch.tensor([1280.], dtype=torch.float32)
        args = (x, ve, pos, x.clone(), x.clone(), x.clone())
        expected = baseline(*args)[0].clone()
        changed = candidate(*args, divisor)[0].clone()
        check = {"start": start, "end": end, "norm_calls": candidate.norm_calls, **metrics(changed, expected)}
        report["torch_candidate_vs_original"].append(check)
        samples.append((args, divisor, expected))
        print("torch", start, end, check["max_abs"], flush=True)
(OUT / "shard0.json").write_text(json.dumps(report, indent=2) + "\n")
if any(row["max_abs"] != 0 for row in report["torch_candidate_vs_original"]):
    raise RuntimeError("Torch candidate changes reference shard; evidence preserved, conversion stopped")

for name, value in candidate.named_buffers():
    if "_cache_" in name:
        value.zero_()
seq = ct.RangeDim(1, 32, default=3)
def tensor(name, shape, dtype=np.float32):
    return ct.TensorType(name=name, shape=shape, dtype=dtype)
inputs = [tensor("x", (1, seq, 1280)), tensor("ve", (1, seq, 2, 128)), tensor("positions", (seq,), np.int32)]
inputs += [tensor(name, (1, seq, 1280)) for name in ("tap0", "tap12", "tap23")]
inputs.append(tensor("divisor1280", (1,)))
sample_x = torch.randn(1, 3, 1280) * .1
trace_input = (sample_x, torch.zeros(1, 3, 2, 128), torch.arange(3, dtype=torch.int32),
               sample_x.clone(), sample_x.clone(), sample_x.clone(), torch.tensor([1280.]))
with torch.inference_mode():
    traced = torch.jit.trace(candidate, trace_input, check_trace=False)
states = [ct.StateType(name=name, wrapped_type=ct.TensorType(shape=value.shape, dtype=np.float16))
          for name, value in candidate.named_buffers() if "_cache_" in name]
t = time.time()
converted = ct.convert(
    traced, inputs=inputs, outputs=[tensor("y", None), tensor("tap23_out", None)], states=states,
    convert_to="mlprogram", compute_precision=ct.precision.FLOAT32,
    minimum_deployment_target=ct.target.macOS15, skip_model_load=True,
)
converted.save(str(PACKAGE))
report["conversion_seconds"] = time.time() - t
report["package_files_sha256"] = {str(path.relative_to(PACKAGE)): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in sorted(PACKAGE.rglob("*")) if path.is_file()}
(OUT / "shard0.json").write_text(json.dumps(report, indent=2) + "\n")
for unit in (ct.ComputeUnit.CPU_ONLY, ct.ComputeUnit.CPU_AND_GPU):
    cm = ct.models.MLModel(str(PACKAGE), compute_units=unit)
    state = cm.make_state()
    report["coreml_vs_original_torch"][unit.name] = []
    for args, divisor, expected in samples:
        payload = {name: value.numpy() for name, value in zip(("x", "ve", "positions", "tap0", "tap12", "tap23"), args)}
        payload["divisor1280"] = divisor.numpy()
        t = time.time()
        got = cm.predict(payload, state=state)["y"]
        span = (int(args[2][0]), int(args[2][-1]) + 1)
        result = {"start": span[0], "end": span[1], "seconds": time.time() - t,
                  **metrics(torch.from_numpy(got), expected)}
        report["coreml_vs_original_torch"][unit.name].append(result)
        print(unit.name, result, flush=True)
        (OUT / "shard0.json").write_text(json.dumps(report, indent=2) + "\n")
