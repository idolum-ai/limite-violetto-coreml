"""Check whether Core ML preserves the source-equivalent sequential 128-product dot."""
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
    p.error("invalid attempt")
OUT = ROOT / "evidence/04d-gpu" / a.attempt
ART = ROOT / "artifacts/04d-gpu" / a.attempt
if OUT.exists() or ART.exists():
    p.error("attempt already exists")

import coremltools as ct
import numpy as np
import torch
from torch import nn
import coreml_port.graphs as graphs
from coreml_port.prediction_guard import install as install_guard

install_guard()
torch.set_num_threads(4)
OUT.mkdir(parents=True)
ART.mkdir(parents=True)
REF = ROOT / "evidence/04d-gpu/candidate4-tiny3-layer38-isolate/reference.npz"
saved = dict(np.load(REF))
q = torch.from_numpy(saved["q"])
key = torch.from_numpy(saved["key"])
value = torch.from_numpy(saved["val"])

class SequentialScores(nn.Module):
    def forward(self, query, keys):
        # Source q and k are Q-rounded BF16 values in FP32 containers, so every
        # product is exactly representable before sequential FP32 accumulation.
        total = query[..., 0].unsqueeze(2) * keys[..., 0].unsqueeze(1)
        for i in range(1, 128):
            total = total + query[..., i].unsqueeze(2) * keys[..., i].unsqueeze(1)
        return total.permute(0, 3, 1, 2) * .1

def metric(actual, want):
    delta = np.abs(actual.astype(np.float64) - want.astype(np.float64))
    bad = np.argwhere(delta != 0)
    return {"different_elements": int(len(bad)), "max_abs": float(delta.max()),
            "first_differences": [{"index": i.tolist(), "actual": float(actual[tuple(i)]),
                                   "expected": float(want[tuple(i)])} for i in bad[:8]]}

report = {"status": "starting", "reference_sha256": hashlib.sha256(REF.read_bytes()).hexdigest(),
          "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "runs": {}}
def save():
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
save()
arrays = {}
mask = torch.from_numpy(saved["mask"])

with torch.inference_mode():
    for count in (3, 2048):
        keys = key[:, :count]
        vals = value[:, :, :count]
        module = SequentialScores().eval()
        expected = module(q, keys).numpy()
        reference = saved["scores"][..., :count]
        traced = torch.jit.trace(module, (q, keys), check_trace=False)
        converted = ct.convert(traced,
            inputs=[ct.TensorType(name="query", shape=q.shape, dtype=np.float32),
                    ct.TensorType(name="keys", shape=keys.shape, dtype=np.float32)],
            outputs=[ct.TensorType(name="scores", dtype=np.float32)],
            compute_precision=ct.precision.FLOAT32,
            minimum_deployment_target=ct.target.macOS15, skip_model_load=True)
        package = ART / f"sequential_q3_k{count}.mlpackage"
        converted.save(str(package))
        cm = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_ONLY)
        actual = cm.predict({"query": q.numpy(), "keys": keys.numpy()})["scores"]

        def rounded(score):
            logits = torch.from_numpy(score).masked_fill(~mask[:, :count][None, None], float("-inf"))
            weights = torch.softmax(logits, -1)
            raw = (weights @ vals).permute(0, 2, 1, 3)
            return graphs.Q(raw).numpy()

        actual_q = rounded(actual)
        expected_q = rounded(expected)
        report["runs"][str(count)] = {
            "torch_sequential_vs_torch_einsum_scores": metric(expected, reference),
            "coreml_sequential_vs_torch_einsum_scores": metric(actual, reference),
            "coreml_vs_torch_sequential_scores": metric(actual, expected),
            "torch_sequential_q_vs_saved": metric(expected_q, saved["rounded"]),
            "coreml_sequential_q_vs_saved": metric(actual_q, saved["rounded"]),
            "target": {"coreml_score": actual[0, 8, 2, :3].tolist(),
                       "torch_score": reference[0, 8, 2, :3].tolist(),
                       "coreml_q": float(actual_q[0, 2, 8, 38]),
                       "saved_q": float(saved["rounded"][0, 2, 8, 38])}}
        arrays[f"k{count}_scores_coreml"] = actual
        arrays[f"k{count}_scores_torch"] = expected
        arrays[f"k{count}_q_coreml"] = actual_q
        save()
report["status"] = "complete"
save()
np.savez_compressed(OUT / "outputs.npz", **arrays)
