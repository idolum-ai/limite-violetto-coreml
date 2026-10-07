"""Prompt-2 layer-2 attention isolation using corrected Torch operands."""
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

OUT = ROOT / "evidence/04d-gpu/integrated-shard/attention"
ART = ROOT / "artifacts/04d-gpu/integrated-shard/attention"
OUT.mkdir(parents=True, exist_ok=False)
ART.mkdir(parents=True, exist_ok=False)

model = load_model()
case = json.loads((ROOT / "evidence/01-reference/local-input-validation.json").read_text())["cases"][2]
ids = torch.tensor([case["input_ids"][:32]], dtype=torch.int32)
divisor = torch.tensor([1280.])
with torch.inference_mode():
    x, ve = NormWrapped(graphs.Embedding(model).eval())(ids, divisor)

captured = {}
original_finish = graphs.Layer.finish


def intercept(self, x, residual, q, keys, values, v, positions, slots):
    if self.i == 2:
        captured.update(
            q=q.permute(0, 2, 1, 3).clone(),
            k=keys.repeat_interleave(5, dim=2).permute(0, 2, 3, 1).clone(),
            val=values.repeat_interleave(5, dim=2).permute(0, 2, 1, 3).clone(),
            mask=(slots[None, :] <= positions[:, None])[None, None].clone(),
        )
    return original_finish(self, x, residual, q, keys, values, v, positions, slots)


graphs.Layer.finish = intercept
try:
    with torch.inference_mode():
        NormWrapped(graphs.Group(model, 0).eval())(
            x, ve, torch.arange(32, dtype=torch.int32), x.clone(), x.clone(), x.clone(), divisor
        )
finally:
    graphs.Layer.finish = original_finish

q, k, val, mask = (captured[n] for n in ("q", "k", "val", "mask"))
# Check that removing permanently masked cache slots leaves this row's source
# operands unchanged. This keeps the isolated compiler experiment small.
q = q[:, 5:6, 31:32, :]
k = k[:, 5:6, :, :]
val = val[:, 5:6, :, :]
mask = mask[:, :, 31:32, :]
with torch.inference_mode():
    full_scores = (q @ k) * .1
    full_weights = softmax(full_scores.masked_fill(~mask, float("-inf")))
    full_raw = full_weights @ val
    prefix_scores = (q @ k[..., :32]) * .1
    prefix_weights = softmax(prefix_scores)
    prefix_raw = prefix_weights @ val[..., :32, :]
prefix_check = {
    "scores_equal": bool(torch.equal(full_scores[..., :32], prefix_scores)),
    "weights_equal": bool(torch.equal(full_weights[..., :32], prefix_weights)),
    "raw_equal": bool(torch.equal(full_raw, prefix_raw)),
    "rounded_equal": bool(torch.equal(graphs.Q(full_raw), graphs.Q(prefix_raw))),
    "full_rounded_element": float(graphs.Q(full_raw)[0, 0, 0, 20]),
    "prefix_rounded_element": float(graphs.Q(prefix_raw)[0, 0, 0, 20]),
}
if not all(prefix_check[k] for k in ("scores_equal", "weights_equal", "raw_equal", "rounded_equal")):
    raise RuntimeError(f"Prefix reduction changes source values: {prefix_check}")
k = k[..., :32].contiguous()
val = val[..., :32, :].contiguous()


class Staged(nn.Module):
    def forward(self, q, k, val):
        dot = q @ k
        scores = dot * .1
        shifted = scores - scores.max(-1, keepdim=True).values
        ex = shifted.exp()
        total = ex[..., :4]
        for j in range(4, ex.shape[-1], 4):
            total = total + ex[..., j:j + 4]
        total = (total[..., 0:1] + total[..., 2:3]) + (total[..., 1:2] + total[..., 3:4])
        inv = torch.ones_like(total) / total
        weights = ex * inv
        raw = weights @ val
        return dot, scores, shifted, ex, total, inv, weights, raw, graphs.Q(raw)


class Product(nn.Module):
    def forward(self, weights, val):
        raw = weights @ val
        return raw, graphs.Q(raw)


report = {
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        ROOT / "coreml_port/graphs.py", ROOT / "coreml_port/ordered_attention.py",
        ROOT / "coreml_port/ordered_linear.py", ROOT / "coreml_port/ordered_norm.py",
        ROOT / "coreml_port/ordered_xsa.py")},
    "compute_units": "CPU_ONLY", "prompt_index": 2, "layer": 2,
    "query_index": 31, "head_index": 5, "value_index": 20,
    "prefix_check": prefix_check, "experiments": {},
}


def run(name, module, args, input_names, output_names):
    with torch.inference_mode():
        expected = module(*args)
    traced = torch.jit.trace(module, args, check_trace=False)
    converted = ct.convert(
        traced,
        inputs=[ct.TensorType(name=n, shape=t.shape, dtype=np.float32) for n, t in zip(input_names, args)],
        outputs=[ct.TensorType(name=n, dtype=np.float32) for n in output_names],
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.macOS15,
        skip_model_load=True,
    )
    package = ART / (name + ".mlpackage")
    converted.save(str(package))
    cm = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_ONLY)
    got = cm.predict({n: t.numpy() for n, t in zip(input_names, args)})
    rows = []
    arrays = {}
    for requested, actual_name, want in zip(output_names, [o.name for o in cm.get_spec().description.output], expected):
        actual = got[actual_name]
        target = want.numpy()
        diff = np.abs(actual.astype(np.float64) - target.astype(np.float64))
        indexes = np.argwhere(diff != 0)
        row = {
            "name": requested, "different_elements": int(len(indexes)),
            "max_abs": float(diff.max()),
            "first_differences": [{"index": ix.tolist(), "actual": float(actual[tuple(ix)]),
                                   "expected": float(target[tuple(ix)])} for ix in indexes[:8]],
        }
        rows.append(row)
        arrays[requested + "_actual"] = actual
        arrays[requested + "_expected"] = target
        print(name, requested, row, flush=True)
    np.savez_compressed(OUT / (name + "-outputs.npz"), **arrays)
    report["experiments"][name] = rows
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return expected


with torch.inference_mode():
    result = run("staged", Staged(), (q, k, val), ("q", "k", "val"),
                 ("dot", "scores", "shifted", "exp", "total", "inv", "weights", "raw", "rounded"))
    run("product", Product(), (result[6], val), ("weights", "val"), ("raw", "rounded"))
