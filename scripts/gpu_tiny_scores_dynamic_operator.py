"""Convert one dynamic score operator and verify Q3/Q32 branch selection."""
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
from coreml_port import dynamic_queries, tiny_scores
from coreml_port.prediction_guard import install as install_guard

install_guard()
torch.set_num_threads(4)
out = ROOT / "evidence/04d-gpu/tiny-scores-dynamic-operator"
art = ROOT / "artifacts/04d-gpu/tiny-scores-dynamic-operator"
if out.exists() or art.exists():
    raise FileExistsError("fresh output paths required")
out.mkdir(parents=True)
art.mkdir(parents=True)
tiny_ref = ROOT / "evidence/04d-gpu/candidate4-tiny3-layer38-isolate/reference.npz"
long_ref = ROOT / "evidence/04d-gpu/long-layer34/reference.npz"
tiny = dict(np.load(tiny_ref))
long = dict(np.load(long_ref))
q3, k3 = tiny["q"], tiny["key"]
q32 = long["q"].transpose(0, 2, 1, 3).copy()
k32 = long["k"].transpose(0, 3, 1, 2).copy()
assert q3.shape == (1, 3, 10, 128) and q32.shape == (1, 32, 10, 128)
assert k3.shape == k32.shape == (1, 2048, 10, 128)

class Scores(nn.Module):
    def forward(self, query, keys):
        return tiny_scores.scores(query, keys)

def compare(actual, expected):
    delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    bad = np.argwhere(delta != 0)
    return {"different_elements": int(len(bad)), "max_abs": float(delta.max()),
            "first_differences": [{"index": i.tolist(), "actual": float(actual[tuple(i)]),
                                   "expected": float(expected[tuple(i)])} for i in bad[:5]]}

report = {"status": "starting", "source_sha256": {
    str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
    for f in (Path(__file__), ROOT / "coreml_port/tiny_scores.py", tiny_ref, long_ref)},
    "shapes": {"q3": list(q3.shape), "q32": list(q32.shape), "key": list(k3.shape)}}
def save():
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
save()

with torch.inference_mode():
    want3 = Scores()(torch.from_numpy(q3), torch.from_numpy(k3)).numpy()
    want32 = Scores()(torch.from_numpy(q32), torch.from_numpy(k32)).numpy()
    ordinary32 = dynamic_queries.scores(torch.from_numpy(q32), torch.from_numpy(k32)).numpy()
report["torch_q3_active_vs_native"] = compare(want3[..., :3], tiny["scores"][..., :3])
report["torch_q32_vs_ordinary"] = compare(want32, ordinary32)
save()

trace = torch.jit.trace(Scores().eval(), (torch.from_numpy(q3), torch.from_numpy(k3)),
                        check_trace=False)
converted = ct.convert(trace,
    inputs=[ct.TensorType(name="query", shape=(1, ct.RangeDim(1, 32, default=3), 10, 128),
                          dtype=np.float32),
            ct.TensorType(name="keys", shape=k3.shape, dtype=np.float32)],
    outputs=[ct.TensorType(name="scores", dtype=np.float32)],
    compute_precision=ct.precision.FLOAT32,
    minimum_deployment_target=ct.target.macOS15,
    skip_model_load=True)
package = art / "scores.mlpackage"
converted.save(str(package))
cm = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_ONLY)
got3 = cm.predict({"query": q3, "keys": k3})["scores"]
got32 = cm.predict({"query": q32, "keys": k32})["scores"]
report["coreml_q3_active_vs_native"] = compare(got3[..., :3], tiny["scores"][..., :3])
report["coreml_q3_vs_torch_branch"] = compare(got3, want3)
report["coreml_q32_vs_ordinary"] = compare(got32, ordinary32)
report["coreml_q32_vs_torch_branch"] = compare(got32, want32)
report["status"] = "complete"
save()
np.savez_compressed(out / "outputs.npz", q3=got3, q32=got32,
                    q3_torch=want3, q32_torch=want32)
