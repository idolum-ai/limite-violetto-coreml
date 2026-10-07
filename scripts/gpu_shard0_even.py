"""Convert and test shard 0 with explicit round-to-even at layer-0 key only."""
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

import coreml_port.graphs as graphs
from coreml_port.cleanup_guard import install
from coreml_port.ordered_linear import linear
from portable.model import load_model
from scripts.gpu_even_round import Q_even
from scripts.probe_full_model import metrics

install()
torch.set_num_threads(4)
graphs.linear = linear
ART = ROOT / "artifacts/04d-gpu"
OUT = ROOT / "evidence/04d-gpu"
ART.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
PACKAGE = ART / "shard0-even.mlpackage"
if PACKAGE.exists():
    raise RuntimeError(f"Refusing to reuse stale package: {PACKAGE}")


def project_key_even(self, x, ve, cos, sin):
    q = graphs.Q(graphs.linear(x, self.q)).reshape(1, -1, 10, 128)
    k = Q_even(graphs.linear(x, self.k)).reshape(1, -1, 2, 128)
    v = graphs.Q(graphs.linear(x, self.v)).reshape(1, -1, 2, 128)
    if self.ve:
        gate = graphs.Q(2 * graphs.Q(torch.sigmoid(graphs.Q(graphs.linear(x[..., :12], self.vg)))))
        v = graphs.Q(v + graphs.Q(gate.unsqueeze(-1) * ve))
    q, k = graphs.norm(q), graphs.norm(k)
    if self.rope:
        q = graphs.Q(graphs.Q(cos * q) + graphs.Q(sin * q.reshape(1, -1, 10, 64, 2).flip(-1).reshape(1, -1, 10, 128)))
        k = graphs.Q(graphs.Q(cos * k) + graphs.Q(sin * k.reshape(1, -1, 2, 64, 2).flip(-1).reshape(1, -1, 2, 128)))
    return q, k, v


def clear_caches(g):
    for name, value in g.named_buffers():
        if "_cache_" in name:
            value.zero_()


ids = json.loads((ROOT / "evidence/01-reference/local-input-validation.json").read_text())["cases"][0]["input_ids"]
model = load_model()
embedding = graphs.Embedding(model).eval()
base = graphs.Group(model, 0).eval()
candidate = graphs.Group(model, 0).eval()
candidate.layers[0].project = types.MethodType(project_key_even, candidate.layers[0])
del model
gc.collect()
report = {
    "created_at": time.time(),
    "sources": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        ROOT / "scripts/gpu_shard0_even.py", ROOT / "scripts/gpu_even_round.py",
        ROOT / "coreml_port/graphs.py", ROOT / "coreml_port/ordered_linear.py")},
    "torch_candidate_vs_original": [],
    "coreml_vs_original_torch": {},
}

samples = []
with torch.inference_mode():
    for start, end in ((0, 32), (32, 33), (33, 36)):
        x, ve = embedding(torch.tensor([ids[start:end]]))
        pos = torch.arange(start, end, dtype=torch.int32)
        inputs = (x, ve, pos, x.clone(), x.clone(), x.clone())
        original = base(*inputs)[0].clone()
        changed = candidate(*inputs)[0].clone()
        check = metrics(changed, original)
        report["torch_candidate_vs_original"].append({"start": start, "end": end, **check})
        samples.append((inputs, original))
        print("torch", start, end, check, flush=True)
(OUT / "shard0-even.json").write_text(json.dumps(report, indent=2) + "\n")

clear_caches(candidate)
seq = ct.RangeDim(1, 32, default=3)
def tensor(name, shape, dtype=np.float32):
    return ct.TensorType(name=name, shape=shape, dtype=dtype)
shape = (1, seq, 1280)
inputs_spec = [tensor("x", shape), tensor("ve", (1, seq, 2, 128)), tensor("positions", (seq,), np.int32)]
inputs_spec += [tensor(name, shape) for name in ("tap0", "tap12", "tap23")]
sample_x = torch.randn(1, 3, 1280) * .1
sample = (sample_x, torch.zeros(1, 3, 2, 128), torch.arange(3, dtype=torch.int32), sample_x.clone(), sample_x.clone(), sample_x.clone())
with torch.inference_mode():
    traced = torch.jit.trace(candidate, sample, check_trace=False)
states = [ct.StateType(name=name, wrapped_type=ct.TensorType(shape=value.shape, dtype=np.float16))
          for name, value in candidate.named_buffers() if "_cache_" in name]
start_time = time.time()
converted = ct.convert(
    traced, inputs=inputs_spec, outputs=[tensor("y", None), tensor("tap23_out", None)],
    states=states, convert_to="mlprogram", compute_precision=ct.precision.FLOAT32,
    minimum_deployment_target=ct.target.macOS15, skip_model_load=True,
)
converted.save(str(PACKAGE))
report["conversion_seconds"] = time.time() - start_time
report["package_files_sha256"] = {
    str(path.relative_to(PACKAGE)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(PACKAGE.rglob("*")) if path.is_file()
}
(OUT / "shard0-even.json").write_text(json.dumps(report, indent=2) + "\n")

for unit in (ct.ComputeUnit.CPU_ONLY, ct.ComputeUnit.CPU_AND_GPU):
    cm = ct.models.MLModel(str(PACKAGE), compute_units=unit)
    state = cm.make_state()
    report["coreml_vs_original_torch"][unit.name] = []
    for (input_values, expected), (start, end) in zip(samples, ((0, 32), (32, 33), (33, 36))):
        payload = {name: value.numpy() for name, value in zip(("x", "ve", "positions", "tap0", "tap12", "tap23"), input_values)}
        t = time.time()
        got = cm.predict(payload, state=state)["y"]
        result = {"start": start, "end": end, "seconds": time.time() - t,
                  **metrics(torch.from_numpy(got), expected)}
        report["coreml_vs_original_torch"][unit.name].append(result)
        print(unit.name, result, flush=True)
        (OUT / "shard0-even.json").write_text(json.dumps(report, indent=2) + "\n")
