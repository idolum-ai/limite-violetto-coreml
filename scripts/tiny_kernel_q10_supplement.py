"""Supplement the saved dynamic Core ML score package with Q10."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import coremltools as ct
import numpy as np
import torch
from coreml_port import tiny_native_scores
from coreml_port.prediction_guard import install


OUT = ROOT / "evidence/04f-tiny/q10-supplement"
OUT.mkdir(parents=True, exist_ok=False)
PACKAGE = ROOT / "artifacts/04f-tiny/dynamic-operator-1/scores.mlpackage"
guard = install()
torch.set_num_threads(4)
seed = 8803
torch.manual_seed(seed + 10 * 100000)
q = torch.randn(1, 10, 10, 128).to(torch.bfloat16).float().contiguous()
k2 = torch.randn(1, 10, 2, 128).to(torch.bfloat16).float().contiguous()
k = k2.repeat_interleave(5, dim=2).contiguous()
keys = torch.zeros((1, 2048, 10, 128), dtype=torch.float32)
keys[:, :10] = k
with torch.inference_mode():
    native = (torch.einsum("bthd,bshd->bhts", q, k) * .1).numpy()
    want = tiny_native_scores.scores(q, keys).numpy()
model = ct.models.MLModel(str(PACKAGE), compute_units=ct.ComputeUnit.CPU_ONLY)
got = model.predict({"query": q.numpy(), "keys": keys.numpy()})["scores"]


def metric(actual, expected):
    diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return {"different": int(np.count_nonzero(diff)), "max_abs": float(diff.max())}


report = {"status": "complete", "q": 10, "seed": seed, "prediction_guard": guard,
          "compute": "CPU_ONLY", "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "module_sha256": hashlib.sha256((ROOT / "coreml_port/tiny_native_scores.py").read_bytes()).hexdigest(),
          "package_files": {str(p.relative_to(PACKAGE)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(PACKAGE.rglob("*")) if p.is_file()},
          "coreml_vs_native_active": metric(got[..., :10], native),
          "coreml_vs_module_full": metric(got, want),
          "shape": list(got.shape)}
report["pass"] = report["coreml_vs_native_active"]["different"] == 0 and \
                 report["coreml_vs_module_full"]["different"] == 0
(OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"pass": report["pass"], "active": report["coreml_vs_native_active"],
                  "full": report["coreml_vs_module_full"]}), flush=True)
raise SystemExit(not report["pass"])
