"""Isolate full-shape layer-2 weighted-value matmul with Torch weights."""
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
OUT = ROOT / "evidence/04d-gpu/integrated-shard/full-product"
ART = ROOT / "artifacts/04d-gpu/integrated-shard/full-product"
OUT.mkdir(parents=True, exist_ok=False)
ART.mkdir(parents=True, exist_ok=False)

model = load_model()
case = json.loads((ROOT / "evidence/01-reference/local-input-validation.json").read_text())["cases"][2]
ids = torch.tensor([case["input_ids"][:32]], dtype=torch.int32)
divisor = torch.tensor([1280.])
with torch.inference_mode():
    x, ve = NormWrapped(graphs.Embedding(model).eval())(ids, divisor)
saved = {}
original_finish = graphs.Layer.finish

def intercept(self, x, residual, q, keys, values, v, positions, slots):
    if self.i == 2:
        saved.update(q=q.permute(0, 2, 1, 3).clone(),
                     k=keys.repeat_interleave(5, dim=2).permute(0, 2, 3, 1).clone(),
                     val=values.repeat_interleave(5, dim=2).permute(0, 2, 1, 3).clone(),
                     mask=(slots[None, :] <= positions[:, None])[None, None].clone())
    return original_finish(self, x, residual, q, keys, values, v, positions, slots)

graphs.Layer.finish = intercept
try:
    with torch.inference_mode():
        NormWrapped(graphs.Group(model, 0).eval())(
            x, ve, torch.arange(32, dtype=torch.int32), x.clone(), x.clone(), x.clone(), divisor)
finally:
    graphs.Layer.finish = original_finish

with torch.inference_mode():
    scores = (saved["q"] @ saved["k"]) * .1
    weights = softmax(scores.masked_fill(~saved["mask"], float("-inf"))).contiguous()
    val = saved["val"].contiguous()
    want_raw = weights @ val
    want_rounded = graphs.Q(want_raw)

class Product(nn.Module):
    def forward(self, weights, val):
        raw = weights @ val
        return raw, graphs.Q(raw)

args = (weights, val)
trace = torch.jit.trace(Product().eval(), args, check_trace=False)
converted = ct.convert(trace,
    inputs=[ct.TensorType(name=n, shape=v.shape, dtype=np.float32) for n, v in zip(("weights", "val"), args)],
    outputs=[ct.TensorType(name="raw", dtype=np.float32), ct.TensorType(name="rounded", dtype=np.float32)],
    compute_precision=ct.precision.FLOAT32, minimum_deployment_target=ct.target.macOS15, skip_model_load=True)
path = ART / "product.mlpackage"
converted.save(str(path))
cm = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_ONLY)
got = cm.predict({n: v.numpy() for n, v in zip(("weights", "val"), args)})
rows = []
arrays = {}
for name, expected in (("raw", want_raw), ("rounded", want_rounded)):
    actual = got[name]
    target = expected.numpy()
    difference = np.abs(actual.astype(np.float64) - target.astype(np.float64))
    ix = np.argwhere(difference != 0)
    rows.append({"name": name, "different_elements": int(len(ix)), "max_abs": float(difference.max()),
                 "problem_element": {"actual": float(actual[0, 5, 31, 20]),
                                     "expected": float(target[0, 5, 31, 20])},
                 "first_differences": [{"index": i.tolist(), "actual": float(actual[tuple(i)]),
                                        "expected": float(target[tuple(i)])} for i in ix[:8]]})
    arrays[name + "_actual"] = actual
    arrays[name + "_expected"] = target
report = {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
              ROOT / "coreml_port/graphs.py", ROOT / "coreml_port/ordered_attention.py",
              ROOT / "coreml_port/ordered_linear.py", ROOT / "coreml_port/ordered_norm.py",
              ROOT / "coreml_port/ordered_xsa.py")},
          "compute_units": "CPU_ONLY", "prompt_index": 2, "layer": 2,
          "weights_shape": list(weights.shape), "value_shape": list(val.shape), "rows": rows}
(OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
np.savez_compressed(OUT / "outputs.npz", **arrays)
print(rows, flush=True)
