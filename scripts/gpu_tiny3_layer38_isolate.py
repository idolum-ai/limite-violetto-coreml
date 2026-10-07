"""Isolate the first candidate-4 tiny-three attention midpoint on saved group-9 input."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

p = argparse.ArgumentParser()
p.add_argument("--attempt", required=True)
a = p.parse_args()
if Path(a.attempt).name != a.attempt or a.attempt in (".", ".."):
    p.error("attempt must be one directory name")
OUT = ROOT / "evidence/04d-gpu" / a.attempt
ART = ROOT / "artifacts/04d-gpu" / a.attempt
if OUT.exists() or ART.exists():
    p.error("attempt already exists")

import coremltools as ct
import numpy as np
import torch
from torch import nn
import coreml_port.graphs as graphs
from coreml_port import dynamic_queries
from coreml_port.ordered_norm import NormWrapped
from coreml_port.prediction_guard import install as install_guard
from portable.model import load_model
from scripts.audit_torch_candidate import make_candidate

install_guard()
torch.set_num_threads(4)
OUT.mkdir(parents=True)
ART.mkdir(parents=True)
BASE = ROOT / "evidence/04d-gpu/candidate4-tiny3-firstgroup"
payload = dict(np.load(BASE / "first-failure-inputs.npz"))
states = dict(np.load(BASE / "first-failure-state-before.npz"))
names = ("x", "ve", "positions", "tap0", "tap12", "tap23", "norm_divisor")
args = tuple(torch.from_numpy(payload[n]) for n in names)
model = load_model("cpu")
_, groups, _ = make_candidate(model, "fp16", False, "ordered", "ordered", "ordered", "lookup", "source")
graphs.prepare_queries = dynamic_queries.prepare
graphs.attention_scores = dynamic_queries.scores
graphs.attention_product = dynamic_queries.product
group = NormWrapped(groups[9]).eval()
del model, groups
for name, buf in group.named_buffers():
    if "_cache_" in name:
        suffix = name.replace(".", "_").split("_cache_")[-1]
        kind = "k_cache_" if "k_cache_" in name else "v_cache_"
        matches = [v for k, v in states.items() if k.endswith(kind + suffix)]
        assert len(matches) == 1, (name, len(matches))
        buf.copy_(torch.from_numpy(matches[0]))

saved = {}
original = graphs.Layer.finish

def capture(self, x, residual, q, keys, values, v, positions, slots):
    if self.i == 38:
        with torch.inference_mode():
            query, pos = graphs.prepare_queries(q, positions)
            key = keys.repeat_interleave(5, dim=2)
            val = values.repeat_interleave(5, dim=2).permute(0, 2, 1, 3)
            scores = graphs.attention_scores(query, key)
            mask = slots[None, :] <= pos[:, None]
            if self.window >= 0:
                mask = mask & (slots[None, :] >= pos[:, None] - self.window)
            logits = scores.masked_fill(~mask[None, None], float("-inf"))
            weights = graphs.softmax(logits)
            raw = graphs.attention_product(weights, val, q.shape[1])
            rounded = graphs.Q(raw)
            saved.update(q=query.numpy(), key=key.numpy(), val=val.numpy(),
                         scores=scores.numpy(), mask=mask.numpy(), logits=logits.numpy(),
                         weights=weights.numpy(), raw=raw.numpy(), rounded=rounded.numpy(),
                         positions=positions.numpy())
    return original(self, x, residual, q, keys, values, v, positions, slots)

graphs.Layer.finish = capture
try:
    with torch.inference_mode():
        y, _ = group(*args)
finally:
    graphs.Layer.finish = original
assert np.array_equal(y.numpy(), payload["expected"]), "corrected Torch group replay differs from native"
np.savez_compressed(OUT / "reference.npz", **saved)

def metric(actual, want):
    delta = np.abs(actual.astype(np.float64) - want.astype(np.float64))
    bad = np.argwhere(delta != 0)
    target = (0, 2, 8, 38) if actual.ndim == 4 and actual.shape[-1] == 128 and actual.shape[1] == 3 else None
    return {"different_elements": int(len(bad)), "max_abs": float(delta.max()),
            "target_actual": None if target is None else float(actual[target]),
            "target_expected": None if target is None else float(want[target]),
            "first_differences": [{"index": i.tolist(), "actual": float(actual[tuple(i)]),
                                   "expected": float(want[tuple(i)])} for i in bad[:5]]}

report = {"status": "starting", "positions": saved["positions"].tolist(),
          "shapes": {n: list(saved[n].shape) for n in ("q", "key", "val", "scores", "weights")},
          "sha256": {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                     for f in (Path(__file__), BASE / "first-failure-inputs.npz",
                               BASE / "first-failure-state-before.npz", ROOT / "coreml_port/graphs.py",
                               ROOT / "coreml_port/dynamic_queries.py", ROOT / "coreml_port/ordered_attention.py")}}
def save():
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
save()
arrays = {}

def run(module, inputs, outputs, stem):
    vals = tuple(torch.from_numpy(saved[n].copy()) for n in inputs)
    traced = torch.jit.trace(module.eval(), vals, check_trace=False)
    converted = ct.convert(traced,
        inputs=[ct.TensorType(name=n, shape=v.shape, dtype=np.float32) for n, v in zip(inputs, vals)],
        outputs=[ct.TensorType(name=n, dtype=np.float32) for n in outputs],
        compute_precision=ct.precision.FLOAT32, minimum_deployment_target=ct.target.macOS15,
        skip_model_load=True)
    package = ART / (stem + ".mlpackage")
    converted.save(str(package))
    cm = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_ONLY)
    got = cm.predict({n: saved[n] for n in inputs})
    arrays.update({stem + "_" + n: got[n] for n in outputs})
    return got

class Product(nn.Module):
    def forward(self, weights, val):
        raw = graphs.attention_product(weights, val, 3)
        return raw, graphs.Q(raw)

got = run(Product(), ("weights", "val"), ("raw", "rounded"), "product")
report["product_given_identical_weights"] = {n: metric(got[n], saved[n]) for n in ("raw", "rounded")}
save()

class Softmax(nn.Module):
    def forward(self, logits):
        return graphs.softmax(logits)

got = run(Softmax(), ("logits",), ("weights",), "softmax")
report["softmax_given_identical_logits"] = metric(got["weights"], saved["weights"])
with torch.inference_mode():
    mixed = graphs.Q(graphs.attention_product(torch.from_numpy(got["weights"]),
                                             torch.from_numpy(saved["val"]), 3)).numpy()
report["softmax_given_identical_logits"]["mixed_rounded"] = metric(mixed, saved["rounded"])
arrays["softmax_mixed_rounded"] = mixed
save()

class Scores(nn.Module):
    def forward(self, q, key):
        return graphs.attention_scores(q, key)

got = run(Scores(), ("q", "key"), ("scores",), "scores")
report["scores_given_identical_qk"] = metric(got["scores"], saved["scores"])
with torch.inference_mode():
    logits = torch.from_numpy(got["scores"]).masked_fill(
        ~torch.from_numpy(saved["mask"])[None, None], float("-inf"))
    mixed_weights = graphs.softmax(logits)
    mixed = graphs.Q(graphs.attention_product(mixed_weights, torch.from_numpy(saved["val"]), 3)).numpy()
report["scores_given_identical_qk"]["mixed_rounded"] = metric(mixed, saved["rounded"])
arrays["scores_mixed_rounded"] = mixed
report["status"] = "complete"
save()
np.savez_compressed(OUT / "outputs.npz", **arrays)
