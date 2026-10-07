"""Check opt-in short score module against frozen map and fresh native dots."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from coreml_port import tiny_native_scores


OUT = ROOT / "evidence/04f-tiny/module-torch"
OUT.mkdir(parents=True, exist_ok=False)
MAP = ROOT / "evidence/04f-tiny/square-candidate-matrix/summary.json"
frozen = json.loads(MAP.read_text())
names = ("sequential", "four_adjacent", "four_shuffle", "q7_tree", "q8_tree")
for case in frozen["cases"]:
    n = case["length"]
    actual = tiny_native_scores.MODE_TABLE[n, :n, :n].numpy()
    expected = np.array([[names.index(x) for x in row] for row in case["strict_map"]])
    assert np.array_equal(actual, expected), n

seeds = (4803, 7207, 9871)
report = {"status": "starting", "seeds": seeds, "torch_version": torch.__version__,
          "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (Path(__file__), ROOT / "coreml_port/tiny_native_scores.py", MAP)},
          "mode_table_matches_frozen_map": True, "cases": []}
torch.set_num_threads(4)
for n in range(1, 16):
    for seed in seeds:
        torch.manual_seed(seed + n * 100000)
        q = torch.randn(1, n, 10, 128).to(torch.bfloat16).float().contiguous()
        k2 = torch.randn(1, n, 2, 128).to(torch.bfloat16).float().contiguous()
        k = k2.repeat_interleave(5, dim=2).contiguous()
        full_keys = torch.zeros((1, 2048, 10, 128), dtype=torch.float32)
        full_keys[:, :n] = k
        with torch.inference_mode():
            native_dot = torch.einsum("bthd,bshd->bhts", q, k).numpy()
            native_score = (torch.from_numpy(native_dot) * .1).numpy()
            raw = tiny_native_scores._short_raw(q, full_keys).numpy()
            score = tiny_native_scores.scores(q, full_keys).numpy()
        raw_wrong = raw[..., :n] != native_dot
        score_wrong = score[..., :n] != native_score
        report["cases"].append({"length": n, "seed": seed,
                                "raw_different": int(np.count_nonzero(raw_wrong)),
                                "scaled_different": int(np.count_nonzero(score_wrong)),
                                "inactive_nonzero": int(np.count_nonzero(score[..., 16:])),
                                "raw_max_abs": float(np.max(np.abs(raw[..., :n].astype(np.float64)-native_dot.astype(np.float64))))})
report["status"] = "complete"
report["all_exact"] = all(c["raw_different"] == c["scaled_different"] == c["inactive_nonzero"] == 0
                          for c in report["cases"])
(OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"status": report["status"], "all_exact": report["all_exact"],
                  "cases": len(report["cases"]), "failures": [c for c in report["cases"]
                                                           if c["raw_different"] or c["scaled_different"] or c["inactive_nonzero"]]}))
