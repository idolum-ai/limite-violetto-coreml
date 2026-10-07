"""Preflight empirical four-lane scores and paired product on native tiny3 boundaries."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import coreml_port.graphs as graphs
from coreml_port import dynamic_queries, paired_product
from scripts import gpu_four_lane_scores
from coreml_port.ordered_norm import NormWrapped
from portable.model import load_model
from scripts.audit_torch_candidate import make_candidate

torch.set_num_threads(4)
p = argparse.ArgumentParser()
p.add_argument("--fold", required=True, choices=("adjacent", "forward"))
a = p.parse_args()
source = ROOT / "evidence/04d-gpu/candidate4-tiny3-firstgroup/native-boundaries.npz"
captures = dict(np.load(source))
out = ROOT / "evidence/04d-gpu" / f"candidate5-tiny3-torch-four-lane-{a.fold}"
if out.exists():
    raise FileExistsError(out)
out.mkdir()

def compare(actual, expected):
    delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    bad = np.argwhere(delta != 0)
    return {"different_elements": int(len(bad)), "max_abs": float(delta.max()),
            "first_differences": [{"index": i.tolist(), "actual": float(actual[tuple(i)]),
                                   "expected": float(expected[tuple(i)])} for i in bad[:5]]}

report = {"status": "starting", "positions": [0, 1, 2], "fold": a.fold,
          "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
          "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "four_lane_scores_sha256": hashlib.sha256((ROOT / "scripts/gpu_four_lane_scores.py").read_bytes()).hexdigest(),
          "paired_product_sha256": hashlib.sha256((ROOT / "coreml_port/paired_product.py").read_bytes()).hexdigest(),
          "groups": []}
def save():
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
save()

model = load_model("cpu")
_, groups, _ = make_candidate(model, "fp16", False, "ordered", "ordered", "ordered", "lookup", "source")
graphs.prepare_queries = dynamic_queries.prepare
graphs.attention_scores = lambda q,k: gpu_four_lane_scores.scores(q,k,a.fold)
graphs.attention_product = paired_product.product

with torch.inference_mode():
    for index, inner in enumerate(groups):
        group = NormWrapped(inner).eval()
        key_in = "embedding_norm" if index == 0 else f"layer_{index * 4 - 1}"
        key_out = "final_hidden" if index == 11 else f"layer_{index * 4 + 3}"
        inputs = (captures[key_in], captures["ve"], np.arange(3, dtype=np.int32),
                  captures["embedding_norm"],
                  captures["layer_11"] if index >= 3 else captures["embedding_norm"],
                  captures["layer_22"] if index >= 6 else captures["embedding_norm"],
                  np.array([1280.], dtype=np.float32))
        got, tap23 = group(*(torch.from_numpy(v) for v in inputs))
        row = {"group": index, "input_reference": key_in, "output_reference": key_out,
               "y": compare(got.numpy(), captures[key_out])}
        if index == 5:
            row["tap23_out"] = compare(tap23.numpy(), captures["layer_22"])
        report["groups"].append(row)
        save()

report["status"] = "complete"
report["all_exact"] = all(row["y"]["different_elements"] == 0 and
                          row.get("tap23_out", {}).get("different_elements", 0) == 0
                          for row in report["groups"])
save()
