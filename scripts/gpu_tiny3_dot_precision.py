"""Compare tiny3 native SGEMM scores with FP64 exact-product accumulation."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from portable.model import load_model

torch.set_num_threads(4)
OUT = ROOT / "evidence/04d-gpu/tiny3-dot-precision"
OUT.mkdir(parents=True, exist_ok=False)
ids = [87555, 130664, 114854]
model = load_model("cpu")
attn = model.backbone.layers[32].self_attn
captured = {}
old = attn.project
def project(hidden, scale, ve, cos, sin):
    result = old(hidden, scale, ve, cos, sin)
    captured["q"] = result[0].detach().clone()
    captured["k"] = result[1].detach().clone()
    captured["q_native_stride"] = list(result[0].stride())
    captured["k_native_stride"] = list(result[1].stride())
    return result
attn.project = project
try:
    with torch.inference_mode():
        model.reference_hidden(torch.tensor([ids], dtype=torch.long))
finally:
    attn.project = old

q = captured["q"].float()
k = captured["k"].repeat_interleave(5, dim=2).float()
native_dot = torch.einsum("bthd,bshd->bhts", q, k).numpy()
native_scores = (torch.from_numpy(native_dot) * .1).numpy()
q64 = q.numpy().astype(np.float64)[0].transpose(1, 0, 2)
k64 = k.numpy().astype(np.float64)[0].transpose(1, 0, 2)
# BF16 operands have at most eight significant bits; 128 exact products and
# their sum fit comfortably in the 53-bit significand of float64.
oracle64 = (q64[:, :, None, :] * k64[:, None, :, :]).sum(axis=-1, dtype=np.float64)[None]
oracle32 = oracle64.astype(np.float32)
oracle_scores_fp32 = np.float32(oracle32 * np.float32(.1))
oracle_scores_fp64 = np.float32(oracle64 * .1)
oracle_scores_fp64_constant32 = np.float32(oracle64 * float(np.float32(.1)))

def compare(actual, expected):
    diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    ix = np.argwhere(diff != 0)
    return {"different_elements": int(len(ix)), "max_abs": float(diff.max()),
            "first_differences": [{"index": i.tolist(), "actual": float(actual[tuple(i)]),
                                   "expected": float(expected[tuple(i)])} for i in ix[:8]]}

report = {"ids": ids, "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
    ROOT / "scripts/gpu_tiny3_dot_precision.py", ROOT / "portable/model.py",
    ROOT / "sources/value-model/modeling_backbone.py", ROOT / "sources/torch-2.7-linear/LinearAlgebra.cpp",
    ROOT / "sources/torch-2.7-linear/CPUBlas.cpp")},
    "operand_layout": {"q_project_shape": list(captured["q"].shape), "q_project_stride": captured["q_native_stride"],
                       "k_project_shape": list(captured["k"].shape), "k_project_stride": captured["k_native_stride"],
                       "q_einsum_shape": list(q.shape), "q_einsum_stride": list(q.stride()),
                       "k_einsum_shape": list(k.shape), "k_einsum_stride": list(k.stride()),
                       "q_as_batched_matrix_shape": list(q.permute(0, 2, 1, 3).reshape(10, 3, 128).shape),
                       "q_as_batched_matrix_stride": list(q.permute(0, 2, 1, 3).reshape(10, 3, 128).stride()),
                       "k_as_batched_matrix_shape": list(k.permute(0, 2, 3, 1).reshape(10, 128, 3).shape),
                       "k_as_batched_matrix_stride": list(k.permute(0, 2, 3, 1).reshape(10, 128, 3).stride())},
    "experiments": {"fp64_sum_cast_fp32_dot": compare(oracle32, native_dot),
                    "fp32_dot_then_fp32_scale": compare(oracle_scores_fp32, native_scores),
                    "fp64_sum_and_real_decimal_scale_then_fp32": compare(oracle_scores_fp64, native_scores),
                    "fp64_sum_and_float32_scale_then_fp32": compare(oracle_scores_fp64_constant32, native_scores)}}
(OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(report["operand_layout"], flush=True)
print(report["experiments"], flush=True)
