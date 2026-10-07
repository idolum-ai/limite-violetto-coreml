"""Validate the frozen Q1..15 score-tree map on independent BF16 inputs."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from sum_topology_reconstruct import evaluate_tree, left_fold


ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "evidence/04f-tiny/square-candidate-matrix/summary.json"
OUT = ROOT / "evidence/04f-tiny/square-holdout"
OUT.mkdir(parents=True, exist_ok=False)
tree_paths = {
    "four_adjacent": ROOT / "evidence/04f-tiny/q3-tail-tree-key0/summary.json",
    "four_shuffle": ROOT / "evidence/04f-tiny/q3-tail-tree-key2/summary.json",
    "q7_tree": ROOT / "evidence/04f-tiny/tree-q7-k7-r4-c0/summary.json",
    "q8_tree": ROOT / "evidence/04f-tiny/tree-q8-k8-r0-c0/summary.json",
}
trees = {"sequential": left_fold(range(128))}
trees.update({name: json.loads(path.read_text())["tree"] for name, path in tree_paths.items()})
frozen = json.loads(MAP.read_text())
assert frozen["status"] == "complete"
seeds = (4803, 7207, 9871)
report = {"status": "starting", "seeds": seeds, "torch_version": torch.__version__,
          "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "frozen_map_sha256": hashlib.sha256(MAP.read_bytes()).hexdigest(),
          "tree_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for name, path in tree_paths.items()}, "cases": []}
torch.set_num_threads(4)
for case in frozen["cases"]:
    length = case["length"]
    assignment = np.array(case["strict_map"])
    assert assignment.shape == (length, length)
    assert not np.isin(assignment, ("+", "-")).any()
    for seed in seeds:
        torch.manual_seed(seed + length * 100000)
        q = torch.randn(1, length, 10, 128).to(torch.bfloat16).float().contiguous()
        k2 = torch.randn(1, length, 2, 128).to(torch.bfloat16).float().contiguous()
        k = k2.repeat_interleave(5, dim=2).contiguous()
        with torch.inference_mode():
            native_dot = torch.einsum("bthd,bshd->bhts", q, k).numpy()[0]
            native_score = (torch.from_numpy(native_dot) * .1).numpy()
        products = (q.numpy()[0, :, None, :, :] * k.numpy()[0, None, :, :, :]).transpose(2, 0, 1, 3)
        pred = np.empty((10, length, length), dtype=np.float32)
        for name in np.unique(assignment):
            coords = np.argwhere(assignment == name)
            vectors = products[:, coords[:, 0], coords[:, 1], :]
            pred[:, coords[:, 0], coords[:, 1]] = evaluate_tree(trees[name], vectors)
        pred_score = pred * np.float32(.1)
        wrong = pred != native_dot
        score_wrong = pred_score != native_score
        report["cases"].append({"length": length, "seed": seed,
                                "raw_different": int(np.count_nonzero(wrong)),
                                "score_different": int(np.count_nonzero(score_wrong)),
                                "raw_max_abs": float(np.max(np.abs(pred.astype(np.float64)-native_dot.astype(np.float64)))),
                                "different_by_query": np.count_nonzero(wrong, axis=(0, 2)).tolist()})
report["status"] = "complete"
report["all_exact"] = all(c["raw_different"] == c["score_different"] == 0 for c in report["cases"])
(OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"status": report["status"], "all_exact": report["all_exact"],
                  "cases": len(report["cases"]), "failures": [c for c in report["cases"] if c["raw_different"] or c["score_different"]]}))
