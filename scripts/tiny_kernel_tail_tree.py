"""Test whether Accelerate's Q3/K3 tail dot is a fixed FP32 sum tree."""
import hashlib
import json
import argparse
from pathlib import Path

import numpy as np

from sum_topology_reconstruct import Oracle, evaluate_tree, reconstruct, validate_triplets
from tiny_kernel_direct_sgemm import direct


ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--key-col", type=int, choices=(0, 1, 2), required=True)
args = parser.parse_args()
OUT = ROOT / f"evidence/04f-tiny/q3-tail-tree-key{args.key_col}"
OUT.mkdir(parents=True, exist_ok=False)
SOURCE = ROOT / "evidence/04d-gpu/bf16-dot-reduction-matrix/operands-and-native.npz"
data = np.load(SOURCE)


def probe(vectors):
    result = np.empty(len(vectors), dtype=np.float32)
    key = np.ones((1, 3, 10, 128), dtype=np.float32)
    for start in range(0, len(vectors), 10):
        chunk = vectors[start:start+10]
        q = np.zeros((1, 3, 10, 128), dtype=np.float32)
        q[0, 2, :len(chunk), :] = chunk
        result[start:start+len(chunk)] = direct(q, key)[0, :len(chunk), 2, args.key_col]
    return result


report = {"status": "starting", "geometry": {"q": 3, "k": 3, "query_row": 2,
                                              "key_col": args.key_col, "batch_heads": 10},
          "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "helper_sha256": hashlib.sha256((ROOT / "scripts/sum_topology_reconstruct.py").read_bytes()).hexdigest(),
          "direct_script_sha256": hashlib.sha256((ROOT / "scripts/tiny_kernel_direct_sgemm.py").read_bytes()).hexdigest(),
          "capture_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest()}
oracle = Oracle(probe)
try:
    tree = reconstruct(oracle)
    report["reconstruction_vectors"] = oracle.vectors_used
    report["reconstruction_calls"] = oracle.calls
    report["held_out_triplets"] = validate_triplets(tree, oracle, count=1000)
    reverse = Oracle(probe, reverse_signs=True)
    report["reverse_signs_triplets"] = validate_triplets(tree, reverse, count=1000, seed=991)
    larger = Oracle(probe, large=2**27)
    report["larger_exponent_triplets"] = validate_triplets(tree, larger, count=1000, seed=997)
    report["saved_random_bf16"] = []
    for seed in (113, 509, 2027):
        q = data[f"q3_seed{seed}"][0, 2]
        k = data[f"k3_seed{seed}"][0, args.key_col]
        products = q * k
        pred = evaluate_tree(tree, products)
        ref = data[f"native_dot_q3_seed{seed}"][0, :, 2, args.key_col]
        report["saved_random_bf16"].append({"seed": seed, "different": int(np.count_nonzero(pred != ref)),
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
