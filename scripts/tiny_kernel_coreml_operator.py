"""Compile and test the opt-in dynamic tiny-score operator on CPU_ONLY."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser()
parser.add_argument("--attempt", required=True)
args = parser.parse_args()
assert args.attempt.isascii() and args.attempt.replace("-", "").isalnum()
OUT = ROOT / "evidence/04f-tiny" / args.attempt
ART = ROOT / "artifacts/04f-tiny" / args.attempt
if OUT.exists() or ART.exists():
    raise FileExistsError("fresh attempt paths required")

import coremltools as ct
import numpy as np
import torch
from torch import nn
from coreml_port import tiny_native_scores
from coreml_port import dynamic_queries
from coreml_port.prediction_guard import install as install_guard

guard = install_guard()
torch.set_num_threads(4)
OUT.mkdir(parents=True)
ART.mkdir(parents=True)
report = {"status": "starting", "attempt": args.attempt, "prediction_guard": guard,
          "torch_version": torch.__version__, "coremltools_version": ct.__version__,
          "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (Path(__file__), ROOT / "coreml_port/tiny_native_scores.py",
                                      ROOT / "coreml_port/dynamic_queries.py",
                                      ROOT / "coreml_port/prediction_guard.py")},
          "cases": []}
def save():
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
save()


def inputs(length, seed=7207):
    torch.manual_seed(seed + length * 100000)
    q = torch.randn(1, length, 10, 128).to(torch.bfloat16).float().contiguous()
    k2 = torch.randn(1, length, 2, 128).to(torch.bfloat16).float().contiguous()
    k = k2.repeat_interleave(5, dim=2).contiguous()
    keys = torch.zeros((1, 2048, 10, 128), dtype=torch.float32)
    keys[:, :length] = k
    return q, k, keys


def metric(actual, expected):
    diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return {"different": int(np.count_nonzero(diff)), "max_abs": float(diff.max())}


class Scores(nn.Module):
    def forward(self, query, keys):
        return tiny_native_scores.scores(query, keys)


q3, _, key3 = inputs(3)
module = Scores().eval()
with torch.inference_mode():
    trace = torch.jit.trace(module, (q3, key3), check_trace=False)
    report["trace_q8_vs_module"] = metric(trace(*inputs(8)[::2]).numpy(),
                                          module(*inputs(8)[::2]).numpy())
    report["trace_q32_vs_module"] = metric(trace(*inputs(32)[::2]).numpy(),
                                           module(*inputs(32)[::2]).numpy())
    assert report["trace_q8_vs_module"]["different"] == 0
    assert report["trace_q32_vs_module"]["different"] == 0
save()
print("trace passed; converting dynamic package", flush=True)
converted = ct.convert(
    trace,
    inputs=[ct.TensorType(name="query", shape=(1, ct.RangeDim(1, 32, default=3), 10, 128),
                          dtype=np.float32),
            ct.TensorType(name="keys", shape=(1, 2048, 10, 128), dtype=np.float32)],
    outputs=[ct.TensorType(name="scores", dtype=np.float32)],
    compute_precision=ct.precision.FLOAT32,
    minimum_deployment_target=ct.target.macOS15,
    skip_model_load=True)
package = ART / "scores.mlpackage"
converted.save(str(package))
report["package"] = str(package.relative_to(ROOT))
report["status"] = "converted"
save()
print("converted; loading CPU_ONLY", flush=True)
model = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_ONLY)
for length in (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 32):
    q, k, keys = inputs(length)
    with torch.inference_mode():
        want = module(q, keys).numpy()
        if length < 16:
            native = (torch.einsum("bthd,bshd->bhts", q, k) * .1).numpy()
            assert np.array_equal(want[..., :length], native)
        else:
            native = dynamic_queries.scores(q, keys).numpy()
            assert np.array_equal(want, native)
    got = model.predict({"query": q.numpy(), "keys": keys.numpy()})["scores"]
    row = {"length": length, "coreml_vs_torch_module": metric(got, want),
           "coreml_active_vs_native": metric(got[..., :length], native[..., :length]),
           "coreml_shape": list(got.shape)}
    report["cases"].append(row)
    save()
    print(f"Q={length}: {row['coreml_active_vs_native']['different']} active score differences", flush=True)
report["status"] = "complete"
report["all_active_exact"] = all(c["coreml_active_vs_native"]["different"] == 0
                                 for c in report["cases"])
report["all_module_exact"] = all(c["coreml_vs_torch_module"]["different"] == 0
                                 for c in report["cases"])
save()
print(json.dumps({"status": report["status"], "all_active_exact": report["all_active_exact"],
                  "all_module_exact": report["all_module_exact"]}), flush=True)
