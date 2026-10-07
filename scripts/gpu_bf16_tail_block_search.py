"""Search contiguous-block FP32 reductions for native odd-query GEMM tail rows."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

source = ROOT / "evidence/04d-gpu/bf16-dot-reduction-matrix/operands-and-native.npz"
data = np.load(source)
out = ROOT / "evidence/04d-gpu/bf16-tail-block-search"
if out.exists():
    raise FileExistsError(out)
out.mkdir(parents=True)

def reduce_values(value, mode):
    if mode == "fp64":
        return value.sum(-1, dtype=np.float64).astype(np.float32)
    if mode == "sequential":
        total = value[..., 0]
        for i in range(1, value.shape[-1]):
            total = np.float32(total + value[..., i])
        return total
    if mode == "reverse":
        total = value[..., -1]
        for i in range(value.shape[-1] - 2, -1, -1):
            total = np.float32(total + value[..., i])
        return total
    a = value.copy()
    while a.shape[-1] > 1:
        if mode == "adjacent":
            a = np.float32(a[..., ::2] + a[..., 1::2])
        else:
            half = a.shape[-1] // 2
            a = np.float32(a[..., :half] + a[..., half:])
    return a[..., 0]

report = {"status": "starting", "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
          "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "cases": {}}
def save():
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
save()

for length, first_tail in ((3, 2), (7, 4)):
    cases = []
    for seed in (113, 509, 2027):
        q = data[f"q{length}_seed{seed}"]
        k = data[f"k{length}_seed{seed}"]
        native = data[f"native_score_q{length}_seed{seed}"][..., first_tail:, :]
        product = np.einsum("bthd,bshd->bhtsd", q, k)[..., first_tail:, :, :]
        cases.append((product, native))
    rows = {}
    for block in (2, 4, 8, 16, 32, 64, 128):
        for inner in ("sequential", "reverse", "adjacent", "halving", "fp64"):
            for outer in ("sequential", "reverse", "adjacent", "halving", "fp64"):
                name = f"block{block}_{inner}_{outer}"
                mismatched = 0
                max_abs = 0.
                for product, native in cases:
                    parts = [reduce_values(product[..., start:start + block], inner)
                             for start in range(0, 128, block)]
                    dot = reduce_values(np.stack(parts, axis=-1), outer)
                    score = dot * np.float32(.1)
                    delta = np.abs(score.astype(np.float64) - native.astype(np.float64))
                    mismatched += int(np.count_nonzero(delta))
                    max_abs = max(max_abs, float(delta.max()))
                rows[name] = {"tail_score_differences": mismatched, "max_abs": max_abs}
    report["cases"][str(length)] = {"tail_start": first_tail, "candidate_count": len(rows),
                                    "candidates": rows,
                                    "best": sorted(rows, key=lambda n: rows[n]["tail_score_differences"])[:12]}
    save()

report["status"] = "complete"
save()
