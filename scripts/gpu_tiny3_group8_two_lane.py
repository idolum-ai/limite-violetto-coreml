"""Replay group 8 tiny3 with empirical two-lane scores plus paired product."""
import argparse
import gc
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
    p.error("invalid attempt")
OUT = ROOT / "evidence/04d-gpu" / a.attempt
ART = ROOT / "artifacts/04d-gpu" / a.attempt
if OUT.exists() or ART.exists():
    p.error("attempt already exists")

import coremltools as ct
import numpy as np
import torch
import coreml_port.graphs as graphs
from coreml_port import dynamic_queries, paired_product
from scripts import gpu_two_lane_scores
from coreml_port.ordered_norm import NormWrapped
from coreml_port.package_identity import verify_candidate_identity
from coreml_port.prediction_guard import install as install_guard
from portable.model import load_model
from scripts.audit_torch_candidate import make_candidate

install_guard()
torch.set_num_threads(4)
identity = verify_candidate_identity(
    ROOT / "artifacts/04d-cpu/full-candidate-4",
    ROOT / "evidence/04d-cpu/full-candidate-4/package-identity.json")
OUT.mkdir(parents=True)
ART.mkdir(parents=True)
BASE = ROOT / "evidence/04d-gpu/candidate4-tiny3-firstgroup"
boundaries = dict(np.load(BASE / "native-boundaries.npz"))
payload = {"x": boundaries["layer_31"], "ve": boundaries["ve"],
           "positions": np.arange(3, dtype=np.int32),
           "tap0": boundaries["embedding_norm"], "tap12": boundaries["layer_11"],
           "tap23": boundaries["layer_22"],
           "norm_divisor": np.array([1280.], dtype=np.float32),
           "expected": boundaries["layer_35"]}
names = ("x", "ve", "positions", "tap0", "tap12", "tap23", "norm_divisor")
args = tuple(torch.from_numpy(payload[n]) for n in names)

model = load_model("cpu")
_, groups, _ = make_candidate(model, "fp16", False, "ordered", "ordered", "ordered", "lookup", "source")
group = NormWrapped(groups[8]).eval()
del groups, model
gc.collect()
graphs.prepare_queries = dynamic_queries.prepare
graphs.attention_product = paired_product.product

graphs.attention_scores = gpu_two_lane_scores.scores

def metric(actual, want):
    delta = np.abs(actual.astype(np.float64) - want.astype(np.float64))
    bad = np.argwhere(delta != 0)
    return {"different_elements": int(len(bad)), "max_abs": float(delta.max()),
            "first_differences": [{"index": i.tolist(), "actual": float(actual[tuple(i)]),
                                   "expected": float(want[tuple(i)])} for i in bad[:8]]}

report = {"status": "starting", "package_identity": identity["aggregate_sha256"],
          "positions": payload["positions"].tolist(),
          "sha256": {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                     for f in (Path(__file__), BASE / "native-boundaries.npz",
                               ROOT / "coreml_port/graphs.py", ROOT / "coreml_port/dynamic_queries.py",
                               ROOT / "scripts/gpu_two_lane_scores.py", ROOT / "coreml_port/paired_product.py")}}
def save():
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
save()

with torch.inference_mode():
    expected_y, expected_tap = group(*args)
report["torch_combined_vs_native"] = metric(expected_y.numpy(), payload["expected"])
save()

# Candidate-4 Core ML group 8 is exact on this native input even though its
# corrected Torch graph is not; preserve this compiled baseline comparison.
baseline = ct.models.MLModel(str(ROOT / "artifacts/04d-cpu/full-candidate-4/8.mlpackage"),
                             compute_units=ct.ComputeUnit.CPU_ONLY)
baseline_got = baseline.predict({n: payload[n] for n in names}, state=baseline.make_state())["y"]
report["candidate4_coreml_vs_native"] = metric(baseline_got, payload["expected"])
save()

traced = torch.jit.trace(group, args, check_trace=False)
seq = ct.RangeDim(1, 32, default=3)
def input_shape(name):
    if name == "positions":
        return (seq,)
    if name in ("x", "tap0", "tap12", "tap23"):
        return (1, seq, 1280)
    if name == "ve":
        return (1, seq, 2, 128)
    return payload[name].shape
converted = ct.convert(traced,
    inputs=[ct.TensorType(name=n, shape=input_shape(n),
                          dtype=np.int32 if n == "positions" else np.float32) for n in names],
    outputs=[ct.TensorType(name="y", dtype=np.float32),
             ct.TensorType(name="tap23_out", dtype=np.float32)],
    states=[ct.StateType(name=n, wrapped_type=ct.TensorType(shape=v.shape, dtype=np.float16))
            for n, v in group.named_buffers() if "_cache_" in n],
    compute_precision=ct.precision.FLOAT32,
    minimum_deployment_target=ct.target.macOS15,
    skip_model_load=True)
package = ART / "group8.mlpackage"
converted.save(str(package))
cm = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_ONLY)
state = cm.make_state()
got = cm.predict({n: payload[n] for n in names}, state=state)
report["coreml_combined_vs_native"] = metric(got["y"], payload["expected"])
report["coreml_combined_vs_torch_combined"] = metric(got["y"], expected_y.numpy())
report["status"] = "complete"
save()
np.savez_compressed(OUT / "outputs.npz", coreml_y=got["y"], torch_y=expected_y.numpy(),
                    native_y=payload["expected"], candidate4_coreml_y=baseline_got)
