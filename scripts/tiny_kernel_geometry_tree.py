"""Recover and test one Accelerate SGEMM output's FP32 reduction topology."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from sum_topology_reconstruct import Oracle, evaluate_tree, reconstruct, validate_triplets
from tiny_kernel_direct_sgemm import direct


parser = argparse.ArgumentParser()
parser.add_argument("--q", type=int, required=True)
parser.add_argument("--k", type=int, required=True)
parser.add_argument("--row", type=int, required=True)
parser.add_argument("--col", type=int, required=True)
args = parser.parse_args()
assert 0 <= args.row < args.q and 0 <= args.col < args.k
assert 128 * args.q * args.k >= 400
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / f"evidence/04f-tiny/tree-q{args.q}-k{args.k}-r{args.row}-c{args.col}"
OUT.mkdir(parents=True, exist_ok=False)
SOURCE = ROOT / "evidence/04d-gpu/bf16-dot-reduction-matrix/operands-and-native.npz"
data = np.load(SOURCE)
key_ones = np.ones((1, args.k, 10, 128), dtype=np.float32)


def probe(vectors):
    result = np.empty(len(vectors), dtype=np.float32)
    for start in range(0, len(vectors), 10):
        chunk = vectors[start:start+10]
        q = np.zeros((1, args.q, 10, 128), dtype=np.float32)
        q[0, args.row, :len(chunk), :] = chunk
        result[start:start+len(chunk)] = direct(q, key_ones)[0, :len(chunk), args.row, args.col]
    return result


report = {"status": "starting", "geometry": {"q": args.q, "k": args.k,
          "query_row": args.row, "key_col": args.col, "heads": 10},
          "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "helper_sha256": hashlib.sha256((ROOT / "scripts/sum_topology_reconstruct.py").read_bytes()).hexdigest(),
          "direct_script_sha256": hashlib.sha256((ROOT / "scripts/tiny_kernel_direct_sgemm.py").read_bytes()).hexdigest(),
          "capture_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
          "operand_range": "BF16-rounded torch.randn, standard normal; finite normal-range values verified"}
oracle = Oracle(probe)
try:
    tree = reconstruct(oracle)
    report["reconstruction_vectors"] = oracle.vectors_used
    report["reconstruction_calls"] = oracle.calls
    report["held_out_triplets"] = validate_triplets(tree, oracle, count=1000)
    report["reverse_signs_triplets"] = validate_triplets(tree, Oracle(probe, reverse_signs=True),
                                                        count=1000, seed=991)
    report["larger_exponent_triplets"] = validate_triplets(tree, Oracle(probe, large=2**27),
                                                          count=1000, seed=997)
    report["fresh_bf16"] = []
    for seed in (381, 727, 1447):
        torch.manual_seed(seed + args.q * 100000 + args.k)
        q = torch.randn(1, args.q, 10, 128).to(torch.bfloat16).float().contiguous()
        k2 = torch.randn(1, args.k, 2, 128).to(torch.bfloat16).float().contiguous()
        k = k2.repeat_interleave(5, dim=2).contiguous()
        with torch.inference_mode():
            native = torch.einsum("bthd,bshd->bhts", q, k).numpy()
        products = q.numpy()[0, args.row] * k.numpy()[0, args.col]
        pred = evaluate_tree(tree, products)
        ref = native[0, :, args.row, args.col]
        report["fresh_bf16"].append({"seed": seed,
                                     "products_all_finite": bool(np.isfinite(products).all()),
                                     "products_min_abs_nonzero": float(np.min(np.abs(products[products != 0]))),
                                     "products_max_abs": float(np.max(np.abs(products))),
                                     "different": int(np.count_nonzero(pred != ref)),
                                     "max_abs": float(np.max(np.abs(pred.astype(np.float64)-ref.astype(np.float64))))})
    report["tree"] = tree
    report["status"] = "complete"
except ValueError as error:
    report["status"] = "binary_tree_rejected"
    report["error"] = str(error)
    report["vectors_used"] = oracle.vectors_used
    report["calls"] = oracle.calls
(OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k:v for k,v in report.items() if k != "tree"}))
