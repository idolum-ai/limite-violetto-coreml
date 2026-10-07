"""Compare explicit FP32 dot reductions against two captured native tiny3 layers."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

torch.set_num_threads(4)
out = ROOT / "evidence/04d-gpu/tiny3-dot-order-contrast"
if out.exists():
    raise FileExistsError(out)
out.mkdir(parents=True)
p32 = ROOT / "evidence/04d-gpu/tiny3-score-candidates/arrays.npz"
p38 = ROOT / "evidence/04d-gpu/candidate4-tiny3-layer38-isolate/reference.npz"
z32 = dict(np.load(p32))
z38 = dict(np.load(p38))

def evaluate(z, layer):
    q = z["q"]
    if layer == 32:
        key, value = z["k"], z["v"]
        source_score, source_q = z["native_score"], z["native_rounded"]
        mask = np.tril(np.ones((3, 3), dtype=np.bool_))
    else:
        key, value = z["key"][:, :3], z["val"][:, :, :3]
        source_score, source_q = z["scores"][..., :3], z["rounded"]
        mask = z["mask"][:, :3]
    product = np.einsum("bthd,bshd->bhtsd", q, key)
    assert product.shape == (1, 10, 3, 3, 128)
    source_score = source_score.copy()

    def row(dot):
        score = dot * np.float32(.1)
        bad = np.argwhere(score != source_score)
        with torch.inference_mode():
            tensor = torch.from_numpy(score)
            m = torch.from_numpy(mask)
            if m.ndim == 2:
                m = m[None]
            weights = torch.softmax(tensor.masked_fill(~m[:, None], float("-inf")), -1)
            if layer == 32:
                raw = torch.einsum("bhts,bshd->bthd", weights, torch.from_numpy(value))
            else:
                raw = (weights @ torch.from_numpy(value)).permute(0, 2, 1, 3)
            rounded = raw.to(torch.bfloat16).float().numpy()
        qbad = np.argwhere(rounded != source_q)
        return {"score_different_elements": int(len(bad)),
                "score_max_abs": float(np.max(np.abs(score.astype(np.float64)-source_score.astype(np.float64)))),
                "rounded_different_elements": int(len(qbad)),
                "rounded_max_abs": float(np.max(np.abs(rounded.astype(np.float64)-source_q.astype(np.float64)))),
                "rounded_first_differences": [
                    {"index": i.tolist(), "actual": float(rounded[tuple(i)]),
                     "native": float(source_q[tuple(i)])} for i in qbad[:3]]}

    results = {}
    for lanes in (1, 2, 4, 8, 16):
        acc = np.zeros(product.shape[:-1] + (lanes,), dtype=np.float32)
        for i in range(128):
            acc[..., i % lanes] = np.float32(acc[..., i % lanes] + product[..., i])
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
                    a = np.float32(a[..., 0::2] + a[..., 1::2])
                dot = a[..., 0]
            else:
                while a.shape[-1] > 1:
                    half = a.shape[-1] // 2
                    a = np.float32(a[..., :half] + a[..., half:])
                dot = a[..., 0]
            results[f"lane{lanes}_{fold}"] = row(dot)
    return results

report = {"status": "complete", "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "capture_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in (p32, p38)},
          "layer32": evaluate(z32, 32), "layer38": evaluate(z38, 38)}
(out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
