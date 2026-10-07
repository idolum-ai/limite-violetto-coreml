"""Probe Torch raw-einsum score changes from FP64 sums and zero-padded GEMM shapes."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

torch.set_num_threads(4)
source = ROOT / "evidence/04d-gpu/bf16-dot-reduction-matrix/operands-and-native.npz"
data = np.load(source)
out = ROOT / "evidence/04d-gpu/bf16-dot-shape-ablation"
if out.exists():
    raise FileExistsError(out)
out.mkdir(parents=True)
report = {"status": "starting", "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
          "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "cases": []}
def save():
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
save()

def compare(score, ref):
    difference = np.abs(score.astype(np.float64) - ref.astype(np.float64))
    bad = np.argwhere(difference != 0)
    rows = np.count_nonzero(difference, axis=(0, 1, 3)).tolist()
    cols = np.count_nonzero(difference, axis=(0, 1, 2)).tolist()
    return {"different_scores": int(len(bad)), "max_abs": float(difference.max()),
            "differences_by_query": rows, "differences_by_key": cols,
            "first_coordinates": [i.tolist() for i in bad[:12]]}

for length in (1, 2, 3, 7, 15):
    for seed in (113, 509, 2027):
        q = torch.from_numpy(data[f"q{length}_seed{seed}"])
        k = torch.from_numpy(data[f"k{length}_seed{seed}"])
        ref = data[f"native_score_q{length}_seed{seed}"]
        qd = q.numpy().astype(np.float64)
        kd = k.numpy().astype(np.float64)
        sum64 = np.einsum("bthd,bshd->bhts", qd, kd)
        score64 = sum64.astype(np.float32) * np.float32(.1)
        row = {"length": length, "seed": seed,
               "fp64_sum_then_fp32_scale": compare(score64, ref), "padding": {}}
        for size in (4, 8, 16):
            if size <= length:
                continue
            for mode in ("q_only", "k_only", "both"):
                qq = q
                kk = k
                if mode in ("q_only", "both"):
                    qq = torch.cat((q, torch.zeros(1, size - length, 10, 128)), dim=1).contiguous()
                if mode in ("k_only", "both"):
                    kk = torch.cat((k, torch.zeros(1, size - length, 10, 128)), dim=1).contiguous()
                with torch.inference_mode():
                    score = (torch.einsum("bthd,bshd->bhts", qq, kk) * .1).numpy()[..., :length, :length]
                row["padding"][f"{mode}_to_{size}"] = compare(score, ref)
        report["cases"].append(row)
        save()

report["status"] = "complete"
save()
