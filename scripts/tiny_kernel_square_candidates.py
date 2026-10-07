"""Classify square short-score geometries against observed reduction trees."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from sum_topology_reconstruct import evaluate_tree, left_fold


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence/04f-tiny/square-candidate-matrix"
OUT.mkdir(parents=True, exist_ok=False)
tree_paths = {
    "four_adjacent": ROOT / "evidence/04f-tiny/q3-tail-tree-key0/summary.json",
    "four_shuffle": ROOT / "evidence/04f-tiny/q3-tail-tree-key2/summary.json",
    "q7_tree": ROOT / "evidence/04f-tiny/tree-q7-k7-r4-c0/summary.json",
    "q8_tree": ROOT / "evidence/04f-tiny/tree-q8-k8-r0-c0/summary.json",
}
trees = {"sequential": left_fold(range(128))}
trees.update({name: json.loads(path.read_text())["tree"] for name, path in tree_paths.items()})
seeds = (381, 727, 1447)
lengths = tuple(range(1, 16))
report = {"status": "starting", "lengths": lengths, "seeds": seeds,
          "candidates": list(trees), "torch_version": torch.__version__,
          "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "tree_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for name, path in tree_paths.items()}, "cases": []}
torch.set_num_threads(4)
for length in lengths:
    common = np.ones((len(trees), 10, length, length), dtype=bool)
    per_seed = []
    for seed in seeds:
        torch.manual_seed(seed + length * 100000)
        q = torch.randn(1, length, 10, 128).to(torch.bfloat16).float().contiguous()
        k2 = torch.randn(1, length, 2, 128).to(torch.bfloat16).float().contiguous()
        k = k2.repeat_interleave(5, dim=2).contiguous()
        with torch.inference_mode():
            ref = torch.einsum("bthd,bshd->bhts", q, k).numpy()[0]
        products = (q.numpy()[0, :, None, :, :] * k.numpy()[0, None, :, :, :]).transpose(2, 0, 1, 3)
        assert products.shape == (10, length, length, 128)
        counts = {}
        for index, (name, tree) in enumerate(trees.items()):
            pred = evaluate_tree(tree, products)
            equal = pred == ref
            common[index] &= equal
            counts[name] = {"different": int(np.count_nonzero(~equal)),
                            "different_by_row": np.count_nonzero(~equal, axis=(0, 2)).tolist()}
        per_seed.append({"seed": seed, "counts": counts})
    compatible_count = common.sum(axis=0)
    no_match = compatible_count == 0
    ambiguous = compatible_count > 1
    strict = common.all(axis=1)
    strict_count = strict.sum(axis=0)
    names = list(trees)
    strict_map = [["-" if strict_count[r, c] == 0 else
                   "+" if strict_count[r, c] > 1 else
                   names[int(np.argmax(strict[:, r, c]))]
                   for c in range(length)] for r in range(length)]
    report["cases"].append({"length": length, "per_seed": per_seed,
                            "no_match": int(no_match.sum()),
                            "no_match_by_row": np.count_nonzero(no_match, axis=(0, 2)).tolist(),
                            "ambiguous": int(ambiguous.sum()),
                            "supported_by_candidate": {name: int(common[i].sum())
                                                       for i, name in enumerate(trees)},
                            "strict_no_match": int(np.count_nonzero(strict_count == 0)),
                            "strict_map": strict_map,
                            "no_match_coordinates_first_20": np.argwhere(no_match)[:20].tolist()})
report["status"] = "complete"
(OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"status": report["status"], "cases": [
    {"length": c["length"], "no_match": c["no_match"],
     "no_match_by_row": c["no_match_by_row"], "ambiguous": c["ambiguous"],
     "strict_no_match": c["strict_no_match"],
     "supported_by_candidate": c["supported_by_candidate"]} for c in report["cases"]]}))
