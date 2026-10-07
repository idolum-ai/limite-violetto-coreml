"""Reproduce Torch's per-head CPU score GEMM through Accelerate's sgemm_."""
import ctypes
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "evidence/04d-gpu/bf16-dot-reduction-matrix/operands-and-native.npz"
OUT = ROOT / "evidence/04f-tiny/direct-sgemm"
torch.set_num_threads(4)

lib = ctypes.CDLL("/System/Library/Frameworks/Accelerate.framework/Accelerate")
sgemm = lib.sgemm_
ptr = ctypes.POINTER(ctypes.c_float)
iptr = ctypes.POINTER(ctypes.c_int)
sgemm.argtypes = [ctypes.c_char_p, ctypes.c_char_p, iptr, iptr, iptr,
                  ptr, ptr, iptr, ptr, iptr, ptr, ptr, iptr]
sgemm.restype = None


def direct(q, key):
    """Same T,N/K,Q,128 call and leading dimensions selected by addmm_impl_cpu_."""
    assert q.dtype == key.dtype == np.float32
    assert q.shape[0] == key.shape[0] == 1 and q.shape[2:] == key.shape[2:] == (10, 128)
    assert q.flags.c_contiguous and key.flags.c_contiguous
    q_length, key_length = q.shape[1], key.shape[1]
    result = np.empty((1, 10, q_length, key_length), dtype=np.float32)
    m, n, k = (ctypes.c_int(x) for x in (key_length, q_length, 128))
    lda, ldb, ldc = (ctypes.c_int(x) for x in
                     (128 if key_length == 1 else 1280,
                      128 if q_length == 1 else 1280, key_length))
    alpha, beta = ctypes.c_float(1), ctypes.c_float(0)
    for h in range(10):
        # The source views keys as [128,K] with strides [1,1280] and queries
        # as [Q,128] with strides [1280,1]. Both keep the interleaved heads.
        a = key[0, :, h, :]
        b = q[0, :, h, :]
        c = result[0, h]
        assert a.strides == (10 * 128 * 4, 4)
        assert b.strides == (10 * 128 * 4, 4)
        assert c.strides == (key_length * 4, 4)
        sgemm(b"T", b"N", ctypes.byref(m), ctypes.byref(n), ctypes.byref(k),
              ctypes.byref(alpha), a.ctypes.data_as(ptr), ctypes.byref(lda),
              b.ctypes.data_as(ptr), ctypes.byref(ldb), ctypes.byref(beta),
              c.ctypes.data_as(ptr), ctypes.byref(ldc))
    return result


def metrics(actual, expected):
    wrong = actual != expected
    return {"different": int(wrong.sum()),
            "different_by_query": wrong.sum(axis=(0, 1, 3)).tolist(),
            "max_abs": float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "summary.json").exists():
        raise FileExistsError(OUT / "summary.json")
    data = np.load(SOURCE)
    report = {"status": "starting", "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "torch_version": torch.__version__,
              "call": {"transa": "T", "transb": "N", "m": "K", "n": "Q", "k": 128,
                       "lda": "1280 (128 if K=1)", "ldb": "1280 (128 if Q=1)",
                       "ldc": "K", "alpha": 1, "beta": 0},
              "cases": []}
    for length in (1, 2, 3, 7, 15):
        for seed in (113, 509, 2027):
            q = data[f"q{length}_seed{seed}"]
            key = data[f"k{length}_seed{seed}"]
            native = data[f"native_dot_q{length}_seed{seed}"]
            actual = direct(q, key)
            report["cases"].append({"length": length, "seed": seed,
                                    "raw_dot": metrics(actual, native),
                                    "scaled_score": metrics(actual * np.float32(.1),
                                                            data[f"native_score_q{length}_seed{seed}"])})
    report["status"] = "complete"
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "cases": report["cases"]}))
