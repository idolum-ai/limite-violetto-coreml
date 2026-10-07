"""Compose short-key sequential score dots in candidate-4 group 9 on saved tiny3 input."""
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
from coreml_port import dynamic_queries
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
payload = dict(np.load(BASE / "first-failure-inputs.npz"))
states = dict(np.load(BASE / "first-failure-state-before.npz"))
names = ("x", "ve", "positions", "tap0", "tap12", "tap23", "norm_divisor")
args = tuple(torch.from_numpy(payload[n]) for n in names)

model = load_model("cpu")
_, groups, _ = make_candidate(model, "fp16", False, "ordered", "ordered", "ordered", "lookup", "source")
group = NormWrapped(groups[9]).eval()
del groups, model
gc.collect()
graphs.prepare_queries = dynamic_queries.prepare
graphs.attention_product = dynamic_queries.product

def sequential_scores(query, keys):
    # Only the first three keys are active on this fixed tiny3 capture. Keep
    # sixteen slots for a bounded short-prefix operator and mask the rest.
    short = keys[:, :16]
    total = query[..., 0].unsqueeze(2) * short[..., 0].unsqueeze(1)
    for i in range(1, 128):
        total = total + query[..., i].unsqueeze(2) * short[..., i].unsqueeze(1)
    active_scores = total.permute(0, 3, 1, 2) * .1
    tail = torch.zeros_like(active_scores[..., :1]).expand(-1, -1, -1, 2032)
    return torch.cat((active_scores, tail), dim=-1)

graphs.attention_scores = sequential_scores
for name, buf in group.named_buffers():
    if "_cache_" in name:
        suffix = name.replace(".", "_").split("_cache_")[-1]
        kind = "k_cache_" if "k_cache_" in name else "v_cache_"
        matches = [v for k, v in states.items() if k.endswith(kind + suffix)]
        assert len(matches) == 1, name
        buf.copy_(torch.from_numpy(matches[0]))

def metric(actual, want):
    delta = np.abs(actual.astype(np.float64) - want.astype(np.float64))
    bad = np.argwhere(delta != 0)
    return {"different_elements": int(len(bad)), "max_abs": float(delta.max()),
            "first_differences": [{"index": i.tolist(), "actual": float(actual[tuple(i)]),
                                   "expected": float(want[tuple(i)])} for i in bad[:8]]}

report = {"status": "starting", "package_identity": identity["aggregate_sha256"],
          "positions": payload["positions"].tolist(),
          "sha256": {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                     for f in (Path(__file__), BASE / "first-failure-inputs.npz",
                               BASE / "first-failure-state-before.npz", ROOT / "coreml_port/graphs.py",
                               ROOT / "coreml_port/dynamic_queries.py")}}
def save():
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
save()

with torch.inference_mode():
    expected_y, expected_tap = group(*args)
report["torch_sequential_vs_native"] = metric(expected_y.numpy(), payload["expected"])
save()
if report["torch_sequential_vs_native"]["different_elements"]:
    report["status"] = "Torch sequential graph diverges; conversion stopped"
    save()
    raise SystemExit(0)

traced = torch.jit.trace(group, args, check_trace=False)
converted = ct.convert(traced,
    inputs=[ct.TensorType(name=n, shape=payload[n].shape,
                          dtype=np.int32 if n == "positions" else np.float32) for n in names],
    outputs=[ct.TensorType(name="y", dtype=np.float32),
             ct.TensorType(name="tap23_out", dtype=np.float32)],
    states=[ct.StateType(name=n, wrapped_type=ct.TensorType(shape=v.shape, dtype=np.float16))
            for n, v in group.named_buffers() if "_cache_" in n],
    compute_precision=ct.precision.FLOAT32,
    minimum_deployment_target=ct.target.macOS15,
    skip_model_load=True)
package = ART / "group9.mlpackage"
converted.save(str(package))
cm = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_ONLY)
state = cm.make_state()
for spec in cm.get_spec().description.state:
    name = spec.name
    suffix = name.replace(".", "_").split("_cache_")[-1]
    kind = "k_cache_" if "k_cache_" in name else "v_cache_"
    matches = [v for k, v in states.items() if k.endswith(kind + suffix)]
    assert len(matches) == 1, name
    state.write_state(name, matches[0])
got = cm.predict({n: payload[n] for n in names}, state=state)
report["coreml_sequential_vs_native"] = metric(got["y"], payload["expected"])
report["coreml_sequential_vs_torch_sequential"] = metric(got["y"], expected_y.numpy())
report["status"] = "complete"
save()
np.savez_compressed(OUT / "outputs.npz", coreml_y=got["y"], torch_y=expected_y.numpy(),
                    native_y=payload["expected"])
