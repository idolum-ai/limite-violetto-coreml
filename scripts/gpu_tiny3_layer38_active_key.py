"""Test exact active-key geometry for the saved tiny3 layer-38 score mismatch."""
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
from coreml_port import dynamic_queries
from coreml_port.prediction_guard import install as install_guard
import coreml_port.graphs as graphs

install_guard()
torch.set_num_threads(4)
OUT.mkdir(parents=True)
ART.mkdir(parents=True)
REF = ROOT / "evidence/04d-gpu/candidate4-tiny3-layer38-isolate/reference.npz"
saved = dict(np.load(REF))
q = torch.from_numpy(saved["q"])
key = torch.from_numpy(saved["key"])
value = torch.from_numpy(saved["val"])

def metric(actual, want):
    delta = np.abs(actual.astype(np.float64) - want.astype(np.float64))
    bad = np.argwhere(delta != 0)
    return {"different_elements": int(len(bad)), "max_abs": float(delta.max()),
            "first_differences": [{"index": i.tolist(), "actual": float(actual[tuple(i)]),
                                   "expected": float(want[tuple(i)])} for i in bad[:8]]}

report = {"status": "starting", "reference_sha256": hashlib.sha256(REF.read_bytes()).hexdigest(),
          "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "positions": saved["positions"].tolist(), "runs": {}}
def save():
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
save()
arrays = {}

class Scores(nn.Module):
    def forward(self, query, keys):
        return dynamic_queries.scores(query, keys)

def cm_scores(query, keys, name):
    traced = torch.jit.trace(Scores().eval(), (query, keys), check_trace=False)
    converted = ct.convert(traced,
        inputs=[ct.TensorType(name="query", shape=query.shape, dtype=np.float32),
                ct.TensorType(name="keys", shape=keys.shape, dtype=np.float32)],
        outputs=[ct.TensorType(name="scores", dtype=np.float32)],
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.macOS15, skip_model_load=True)
    package = ART / (name + ".mlpackage")
    converted.save(str(package))
    cm = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_ONLY)
    return cm.predict({"query": query.numpy(), "keys": keys.numpy()})["scores"]

with torch.inference_mode():
    for key_count in (3, 2048):
        query = q
        keys = key[:, :key_count]
        vals = value[:, :, :key_count]
        expected_scores = dynamic_queries.scores(query, keys).numpy()
        actual_scores = cm_scores(query, keys, f"q3_k{key_count}")
        # Native attention reference uses Torch softmax on exactly the active key span.
        mask = torch.from_numpy(saved["mask"][:, :key_count])
        def rounded(scores):
            logits = torch.from_numpy(scores).masked_fill(~mask[None, None], float("-inf"))
            weights = torch.softmax(logits, -1)
            return graphs.Q((weights @ vals).permute(0, 2, 1, 3)).numpy()
        want_q = rounded(expected_scores)
        actual_q = rounded(actual_scores)
        report["runs"][str(key_count)] = {
            "scores": metric(actual_scores, expected_scores),
            "rounded_given_coreml_scores": metric(actual_q, want_q),
            "torch_rounded_vs_saved_corrected": metric(want_q, saved["rounded"]),
            "coreml_score_rounded_vs_saved_corrected": metric(actual_q, saved["rounded"]),
            "target_scores": {"actual": actual_scores[0, 8, 2, :3].tolist(),
                              "torch": expected_scores[0, 8, 2, :3].tolist()},
            "target_q": {"actual": float(actual_q[0, 2, 8, 38]),
                         "torch": float(want_q[0, 2, 8, 38]),
                         "saved_corrected": float(saved["rounded"][0, 2, 8, 38])}}
        arrays[f"q3_k{key_count}_scores_actual"] = actual_scores
        arrays[f"q3_k{key_count}_scores_torch"] = expected_scores
        arrays[f"q3_k{key_count}_rounded_actual"] = actual_q
        arrays[f"q3_k{key_count}_rounded_torch"] = want_q
        save()

report["status"] = "complete"
save()
np.savez_compressed(OUT / "outputs.npz", **arrays)
