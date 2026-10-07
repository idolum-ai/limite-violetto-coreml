"""Offline FP32 reciprocal substitution on saved warm group-0 normalization taps."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPLAY = ROOT / "evidence/06e-gpu-normalization/replay-1/results.json"
SUMMARY = ROOT / "evidence/06e-gpu-normalization/residual-summary-1.json"
ANALYSIS = ROOT / "evidence/06e-gpu-normalization/normalization-analysis.json"
OUTPUT = ROOT / "evidence/06e-gpu-normalization/normalization-counterfactual.json"
WARM = 1
NAMES = ("value_17", "denom_7", "var_5675", "x_67", "step_47", "var_5690", "var_5691", "x_69")


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(path)
    return result


def archive(meta: dict) -> dict[str, np.ndarray]:
    path = ROOT / meta["path"]
    if sha(path) != meta["sha256"]:
        raise ValueError(f"Archive changed: {path}")
    with np.load(path) as saved:
        result = {name: saved[name].copy() for name in saved.files}
    assert set(result) == set(meta["dtypes"]) == set(meta["shapes"])
    for name, array in result.items():
        assert list(array.shape) == meta["shapes"][name]
        assert str(array.dtype) == meta["dtypes"][name]
        assert np.isfinite(array).all()
    return result


def compare(actual: np.ndarray, reference: np.ndarray) -> dict:
    assert actual.shape == reference.shape and actual.dtype == reference.dtype == np.float32
    mask = np.ascontiguousarray(actual).view(np.uint32) != np.ascontiguousarray(reference).view(np.uint32)
    first = None
    if np.any(mask):
        index = tuple(int(i) for i in np.argwhere(mask)[0])
        first = {"index": list(index), "actual": float(actual[index]),
                 "reference": float(reference[index])}
    rows = np.flatnonzero(np.any(mask, axis=tuple(i for i in range(mask.ndim) if i != 1)))
    return {"bitwise_error_count": int(np.count_nonzero(mask)),
            "numeric_error_count": int(np.count_nonzero(actual != reference)),
            "query_rows_with_differences": [int(i) for i in rows],
            "first_difference": first}


def reconstruct(taps: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    value = taps["value_17"]
    reciprocal = taps["var_5675"]
    step = taps["step_47"]
    x67 = np.multiply(value, reciprocal, dtype=np.float32)
    quotient = np.divide(x67, step, dtype=np.float32)
    rounded = np.rint(quotient).astype(np.float32)
    x69 = np.multiply(rounded, step, dtype=np.float32)
    return {"x_67": x67, "var_5690": quotient, "var_5691": rounded, "x_69": x69}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Fresh counterfactual output required: {OUTPUT}")
    replay, summary, analysis = map(read, (REPLAY, SUMMARY, ANALYSIS))
    assert replay["status"] == "completed" and replay["passed"] and replay["instrumentation_valid"]
    assert summary["controls_passed"] and summary["replay"]["sha256"] == sha(REPLAY)
    assert analysis["controls_passed"] and analysis["sources"]["replay"]["sha256"] == sha(REPLAY)
    assert analysis["sources"]["residual_summary"]["sha256"] == sha(SUMMARY)
    assert analysis["warm_first_differing_operation"] == "var_5675"
    controls = {(row["chunk"], row["policy"]): row for row in replay["controls"]}
    assert controls[(WARM, "CPU")]["passed"] and controls[(WARM, "GPU")]["passed"]
    cpu_meta = replay["files"][f"chunk{WARM}/CPU-tapped/outputs"]
    gpu_meta = replay["files"][f"chunk{WARM}/GPU-tapped/outputs"]
    cpu, gpu = archive(cpu_meta), archive(gpu_meta)
    assert set(NAMES) <= set(cpu) == set(gpu)
    assert compare(gpu["value_17"], cpu["value_17"])["bitwise_error_count"] == 0
    assert compare(gpu["denom_7"], cpu["denom_7"])["bitwise_error_count"] == 0
    assert compare(gpu["step_47"], cpu["step_47"])["bitwise_error_count"] == 0
    assert compare(gpu["var_5675"], cpu["var_5675"])["bitwise_error_count"] == 3
    native_reconstruction = {}
    for policy, taps in (("CPU", cpu), ("GPU", gpu)):
        generated = reconstruct(taps)
        native_reconstruction[policy] = {name: compare(generated[name], taps[name])
                                         for name in generated}
        assert all(item["bitwise_error_count"] == 0 for item in native_reconstruction[policy].values())
    substituted = dict(gpu)
    substituted["var_5675"] = cpu["var_5675"].copy()
    generated = reconstruct(substituted)
    before = {name: compare(gpu[name], cpu[name]) for name in generated}
    after = {name: compare(generated[name], cpu[name]) for name in generated}
    assert before["var_5691"]["bitwise_error_count"] == before["x_69"]["bitwise_error_count"] == 19
    assert all(item["bitwise_error_count"] == 0 for item in after.values())
    result = {"schema_version": 1,
              "script_sha256": sha(Path(__file__)),
              "sources": {"replay": {"path": str(REPLAY.relative_to(ROOT)), "sha256": sha(REPLAY)},
                          "residual_summary": {"path": str(SUMMARY.relative_to(ROOT)), "sha256": sha(SUMMARY)},
                          "normalization_analysis": {"path": str(ANALYSIS.relative_to(ROOT)),
                                                     "sha256": sha(ANALYSIS)}},
              "archives": {"CPU_tapped_outputs_sha256": cpu_meta["sha256"],
                           "GPU_tapped_outputs_sha256": gpu_meta["sha256"]},
              "query_length": int(cpu["value_17"].shape[1]),
              "equal_saved_inputs": {name: compare(gpu[name], cpu[name]) for name in
                                     ("value_17", "denom_7", "step_47")},
              "saved_reciprocal_difference": compare(gpu["var_5675"], cpu["var_5675"]),
              "native_stage_reconstruction": native_reconstruction,
              "original_gpu_vs_cpu": before,
              "offline_gpu_with_cpu_reciprocal_vs_cpu": after,
              "offline_local_path_exact": True,
              "scope": "Offline NumPy FP32 substitution on saved warm taps only. Native CPU/GPU stage values were independently reconstructed bitwise before substitution. This is not a native graph fix, state replay, full-model parity result, or speed claim.",
              "numpy_version": np.__version__, "python_version": sys.version.split()[0]}
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(json.dumps({"offline_local_path_exact": True,
                      "round_differences_before": before["var_5691"]["bitwise_error_count"],
                      "round_differences_after": after["var_5691"]["bitwise_error_count"],
                      "x69_differences_after": after["x_69"]["bitwise_error_count"]}, indent=2))


if __name__ == "__main__":
    main()
