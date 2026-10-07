"""Search reproducible FP32 dot orders against native raw-layout Torch einsum."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

torch.set_num_threads(4)
out = ROOT / "evidence/04d-gpu/bf16-dot-reduction-matrix"
if out.exists():
    raise FileExistsError(out)
out.mkdir(parents=True)
lengths = (1, 2, 3, 7, 15)
seeds = (113, 509, 2027)
report = {"status": "starting", "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "torch_version": torch.__version__, "lengths": lengths, "seeds": seeds,
          "operand_semantics": "BF16-valued FP32, raw [1,time,heads,128] layout; K2 repeated fivefold to H10",
          "cases": []}
def save():
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
save()
arrays = {}

def reductions(products):
    candidates = {}
    for lanes in (1, 2, 4, 8, 16):
        acc = np.zeros(products.shape[:-1] + (lanes,), dtype=np.float32)
        for i in range(128):
            acc[..., i % lanes] = np.float32(acc[..., i % lanes] + products[..., i])
        for fold in ("forward", "reverse", "adjacent_tree", "shuffle_tree"):
            a = acc.copy()
            if fold == "forward":
                dot = a[..., 0]
                for i in range(1, lanes):
                    dot = np.float32(dot + a[..., i])
            elif fold == "reverse":
                dot = a[..., -1]
                for i in range(lanes - 2, -1, -1):
                    dot = np.float32(dot + a[..., i])
            elif fold == "adjacent_tree":
                while a.shape[-1] > 1:
                    a = np.float32(a[..., ::2] + a[..., 1::2])
                dot = a[..., 0]
            else:
                while a.shape[-1] > 1:
                    half = a.shape[-1] // 2
                    a = np.float32(a[..., :half] + a[..., half:])
                dot = a[..., 0]
            candidates[f"lane{lanes}_{fold}"] = dot
    return candidates

for length in lengths:
    for seed in seeds:
        torch.manual_seed(seed + length * 100000)
        q = torch.randn(1, length, 10, 128).to(torch.bfloat16).float().contiguous()
        k2 = torch.randn(1, length, 2, 128).to(torch.bfloat16).float().contiguous()
        k = k2.repeat_interleave(5, dim=2).contiguous()
        assert q.shape == k.shape == (1, length, 10, 128)
        with torch.inference_mode():
            native_dot = torch.einsum("bthd,bshd->bhts", q, k)
            native_score = native_dot * .1
        qn, kn = q.numpy(), k.numpy()
        products = np.einsum("bthd,bshd->bhtsd", qn, kn)
        assert products.shape == (1, 10, length, length, 128)
        dot_ref = native_dot.numpy()
        score_ref = native_score.numpy()
        rows = {}
        for name, dot in reductions(products).items():
            score = dot * np.float32(.1)
            rows[name] = {
                "dot_different_elements": int(np.count_nonzero(dot != dot_ref)),
                "score_different_elements": int(np.count_nonzero(score != score_ref)),
                "score_max_abs": float(np.max(np.abs(score.astype(np.float64)-score_ref.astype(np.float64))))}
        report["cases"].append({"length": length, "seed": seed,
            "contract_work": 128 * length * length,
            "q_stride": list(q.stride()), "k_stride": list(k.stride()),
            "q_head_stride": list(q.permute(0, 2, 1, 3).stride()),
            "k_head_t_stride": list(k.permute(0, 2, 3, 1).stride()),
            "scores": rows})
        arrays[f"q{length}_seed{seed}"] = qn
        arrays[f"k{length}_seed{seed}"] = kn
        arrays[f"native_dot_q{length}_seed{seed}"] = dot_ref
        arrays[f"native_score_q{length}_seed{seed}"] = score_ref
        save()

report["status"] = "complete"
save()
np.savez_compressed(out / "operands-and-native.npz", **arrays)
