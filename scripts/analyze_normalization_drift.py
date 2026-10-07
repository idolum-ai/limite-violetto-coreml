"""Analyze saved normalization taps on equal warm inputs; never runs a model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "evidence/06e-gpu-normalization/protocol.json"
GROUP_REPLAY = ROOT / "evidence/06d-gpu-rounding/group0-replay-1/results.json"
PARENTS = {
    "x_65": ("value_17",),
    "var_5669": ("x_65",),
    "mean_7": ("var_5669", "norm_divisor"),
    "var_5672": ("mean_7", "epsilon_2^-23"),
    "denom_7": ("var_5672",),
    "var_5675": ("denom_7",),
    "x_67": ("value_17", "var_5675"),
    "step_47": ("x_67",),
    "var_5690": ("x_67", "step_47"),
    "var_5691": ("var_5690",),
    "x_69": ("var_5691", "step_47"),
}


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def load(meta: dict) -> dict[str, np.ndarray]:
    path = ROOT / meta["path"]
    if sha(path) != meta["sha256"]:
        raise ValueError(f"Archive changed: {path}")
    with np.load(path) as source:
        arrays = {name: source[name].copy() for name in source.files}
    assert set(arrays) == set(meta["shapes"]) == set(meta["dtypes"])
    for name, array in arrays.items():
        assert list(array.shape) == meta["shapes"][name]
        assert str(array.dtype) == meta["dtypes"][name]
        assert np.isfinite(array).all()
    return arrays


def bits_differ(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    assert a.dtype == b.dtype == np.float32 and a.shape == b.shape
    return np.ascontiguousarray(a).view(np.uint32) != np.ascontiguousarray(b).view(np.uint32)


def metrics(a: np.ndarray, ref: np.ndarray) -> dict:
    changed = bits_differ(a, ref)
    delta = a.astype(np.float64) - ref.astype(np.float64)
    rmse = float(np.sqrt(np.mean(delta * delta)))
    reference_rms = float(np.sqrt(np.mean(ref.astype(np.float64) ** 2)))
    first = None
    if np.any(changed):
        index = tuple(int(i) for i in np.argwhere(changed)[0])
        first = {"index": list(index), "gpu": float(a[index]), "cpu": float(ref[index])}
    rows = np.flatnonzero(np.any(changed, axis=tuple(i for i in range(changed.ndim) if i != 1)))
    return {"bitwise_error_count": int(np.count_nonzero(changed)),
            "numeric_error_count": int(np.count_nonzero(a != ref)),
            "max_abs": float(np.max(np.abs(delta))), "rmse": rmse,
            "relative_rmse": rmse / reference_rms if reference_rms else None,
            "query_rows_with_differences": [int(row) for row in rows],
            "first_difference": first}


def midpoint_analysis(cpu: dict[str, np.ndarray], gpu: dict[str, np.ndarray]) -> dict:
    names = ("x_67", "step_47", "var_5690", "var_5691", "x_69")
    if any(name not in cpu or name not in gpu for name in names):
        return {"available": False, "reason": "Quotient, round, or scaling taps absent"}
    q_cpu = cpu["var_5690"]
    q_gpu = gpu["var_5690"]
    round_cpu = cpu["var_5691"]
    round_gpu = gpu["var_5691"]
    step_cpu = cpu["step_47"]
    step_gpu = gpu["step_47"]
    output_cpu = cpu["x_69"]
    output_gpu = gpu["x_69"]
    assert q_cpu.shape == q_gpu.shape == round_cpu.shape == round_gpu.shape == \
        step_cpu.shape == step_gpu.shape == output_cpu.shape == output_gpu.shape
    low = np.minimum(q_cpu.astype(np.float64), q_gpu.astype(np.float64))
    high = np.maximum(q_cpu.astype(np.float64), q_gpu.astype(np.float64))
    half_boundary = np.ceil(low - 0.5) + 0.5
    quotient_changed = bits_differ(q_gpu, q_cpu)
    rounded_changed = bits_differ(round_gpu, round_cpu)
    rounded_numeric_changed = round_gpu != round_cpu
    step_changed = bits_differ(step_gpu, step_cpu)
    output_changed = bits_differ(output_gpu, output_cpu)
    spanned_or_touched = (low < high) & (low <= half_boundary) & (half_boundary <= high)
    strictly_crossed = (low < half_boundary) & (half_boundary < high)
    cpu_at_half = (q_cpu.astype(np.float64) - np.floor(q_cpu.astype(np.float64))) == 0.5
    gpu_at_half = (q_gpu.astype(np.float64) - np.floor(q_gpu.astype(np.float64))) == 0.5
    examples = []
    for index_array in np.argwhere(rounded_changed)[:12]:
        index = tuple(int(i) for i in index_array)
        examples.append({"index": list(index), "q_cpu": float(q_cpu[index]),
                         "q_gpu": float(q_gpu[index]), "round_cpu": float(round_cpu[index]),
                         "round_gpu": float(round_gpu[index]),
                         "step_cpu": float(step_cpu[index]), "step_gpu": float(step_gpu[index]),
                         "x69_cpu": float(output_cpu[index]), "x69_gpu": float(output_gpu[index]),
                         "half_boundary": float(half_boundary[index]),
                         "half_boundary_spanned_or_touched": bool(spanned_or_touched[index])})
    return {"available": True, "quotient_bitwise_error_count": int(np.count_nonzero(quotient_changed)),
            "round_bitwise_error_count": int(np.count_nonzero(rounded_changed)),
            "round_numeric_error_count": int(np.count_nonzero(rounded_numeric_changed)),
            "step_bitwise_error_count": int(np.count_nonzero(step_changed)),
            "x69_bitwise_error_count": int(np.count_nonzero(output_changed)),
            "half_boundaries_spanned_or_touched": int(np.count_nonzero(spanned_or_touched)),
            "half_boundaries_strictly_crossed": int(np.count_nonzero(strictly_crossed)),
            "round_numeric_changes_with_half_boundary_spanned_or_touched":
                int(np.count_nonzero(rounded_numeric_changed & spanned_or_touched)),
            "round_numeric_changes_without_half_boundary_spanned_or_touched":
                int(np.count_nonzero(rounded_numeric_changed & ~spanned_or_touched)),
            "round_changes_with_equal_quotient_bits":
                int(np.count_nonzero(rounded_changed & ~quotient_changed)),
            "cpu_quotient_exact_half_count": int(np.count_nonzero(cpu_at_half)),
            "gpu_quotient_exact_half_count": int(np.count_nonzero(gpu_at_half)),
            "x69_changes_with_round_numeric_change":
                int(np.count_nonzero(output_changed & rounded_numeric_changed)),
            "x69_changes_with_step_change": int(np.count_nonzero(output_changed & step_changed)),
            "examples": examples,
            "interpretation": "Half-boundary counts identify crossings or touches in the saved quotient arrays; they do not adjudicate CPU or GPU sqrt rounding correctness."}


def reciprocal_analysis(cpu: dict[str, np.ndarray], gpu: dict[str, np.ndarray]) -> dict:
    if "denom_7" not in cpu or "var_5675" not in cpu:
        return {"available": False}
    dc, dg = cpu["denom_7"], gpu["denom_7"]
    rc, rg = cpu["var_5675"], gpu["var_5675"]
    assert dc.shape == dg.shape == rc.shape == rg.shape
    assert dc.dtype == dg.dtype == rc.dtype == rg.dtype == np.float32
    assert np.all(dc > 0) and np.all(dg > 0)
    # The FP32 denominator is the exact input to reciprocal here. Float64
    # division followed by FP32 cast supplies a nearest-FP32 reference; this
    # does not make any assertion about sqrt's correctly rounded result.
    ref_cpu = (1.0 / dc.astype(np.float64)).astype(np.float32)
    ref_gpu = (1.0 / dg.astype(np.float64)).astype(np.float32)
    cpu_ulp = rc.view(np.uint32).astype(np.int64) - ref_cpu.view(np.uint32).astype(np.int64)
    gpu_ulp = rg.view(np.uint32).astype(np.int64) - ref_gpu.view(np.uint32).astype(np.int64)
    equal_denominator = ~bits_differ(dg, dc)
    changed_reciprocal = bits_differ(rg, rc)

    def distribution(errors: np.ndarray) -> dict[str, int]:
        values, counts = np.unique(errors, return_counts=True)
        return {str(int(value)): int(count) for value, count in zip(values, counts)}

    rows = []
    for row in range(dc.shape[1]):
        index = (0, row, 0)
        if changed_reciprocal[index]:
            rows.append({"query_row": row, "denominator_bitwise_equal": bool(equal_denominator[index]),
                         "denominator_cpu": float(dc[index]), "denominator_gpu": float(dg[index]),
                         "reciprocal_cpu": float(rc[index]), "reciprocal_gpu": float(rg[index]),
                         "nearest_fp32_cpu_denominator": float(ref_cpu[index]),
                         "nearest_fp32_gpu_denominator": float(ref_gpu[index]),
                         "cpu_ulp_from_nearest": int(cpu_ulp[index]),
                         "gpu_ulp_from_nearest": int(gpu_ulp[index])})
    return {"available": True, "denominator_equal_count": int(np.count_nonzero(equal_denominator)),
            "denominator_different_count": int(np.count_nonzero(~equal_denominator)),
            "reciprocal_different_count": int(np.count_nonzero(changed_reciprocal)),
            "reciprocal_different_with_equal_denominator_count":
                int(np.count_nonzero(changed_reciprocal & equal_denominator)),
            "cpu_ulp_error_distribution": distribution(cpu_ulp),
            "gpu_ulp_error_distribution": distribution(gpu_ulp),
            "different_rows": rows,
            "reference_scope": "Nearest FP32 reciprocal of each saved FP32 denom_7; no correctness assumption about how denom_7 was computed."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    replay_path = (ROOT / args.replay).resolve()
    summary_path = (ROOT / args.summary).resolve()
    output_path = (ROOT / args.output).resolve()
    if not all(path.is_relative_to(ROOT) for path in (replay_path, summary_path, output_path)):
        raise ValueError("All paths must remain in repository")
    if output_path.exists():
        raise FileExistsError(f"Fresh output required: {output_path}")
    replay, summary, protocol, group = map(read, (replay_path, summary_path, PROTOCOL, GROUP_REPLAY))
    assert replay["status"] == "completed" and replay["passed"] and replay["instrumentation_valid"]
    assert summary["controls_passed"] and summary["replay"]["sha256"] == sha(replay_path)
    assert summary["script_sha256"] == sha(ROOT / "scripts/summarize_rounding_residual.py")
    assert group["status"] == "completed" and group["cpu_preservation_passed"]
    assert replay["pins"][str(GROUP_REPLAY.relative_to(ROOT))] == sha(GROUP_REPLAY)
    for relative, expected in replay["pins"].items():
        assert sha(ROOT / relative) == expected
    ordered = [entry["name"] for entry in protocol["checkpoint_candidates"]]
    assert replay["taps"] == ordered == summary["taps"]
    archives = {name: load(meta) for name, meta in replay["files"].items()}
    rows = []
    for chunk in (0, 1):
        cpu = archives[f"chunk{chunk}/CPU-tapped/outputs"]
        gpu = archives[f"chunk{chunk}/GPU-tapped/outputs"]
        assert set(ordered) <= set(cpu) == set(gpu)
        measured = {name: metrics(gpu[name], cpu[name]) for name in ordered}
        summary_chunk = next(row for row in summary["chunks"] if row["chunk"] == chunk)
        for name in ordered:
            previously = summary_chunk["tap_comparisons"][name]
            for key in ("bitwise_error_count", "numeric_error_count", "query_rows_with_differences"):
                assert measured[name][key] == previously[key]
            for key in ("max_abs", "rmse", "relative_rmse"):
                assert measured[name][key] == previously[key]
        first = next((name for name in ordered if measured[name]["bitwise_error_count"]), None)
        assert first == summary_chunk["first_changed_tap"]
        direct_inputs = list(PARENTS.get(first, ())) if first else []
        direct_input_checks = {name: (True if name in ("norm_divisor", "epsilon_2^-23") else
                                      measured[name]["bitwise_error_count"] == 0)
                               for name in direct_inputs}
        input_exact = measured["value_17"]["bitwise_error_count"] == 0
        rows.append({"chunk": chunk, "query_length": int(cpu[ordered[0]].shape[1]),
                     "value17_exact": input_exact,
                     "first_differing_operation": first,
                     "first_differing_operation_index": next((entry["candidate_operation_index"]
                                                               for entry in protocol["checkpoint_candidates"]
                                                               if entry["name"] == first), None),
                     "first_difference_direct_inputs": direct_input_checks,
                     "first_difference_direct_inputs_exact": all(direct_input_checks.values())
                     if direct_input_checks else None,
                     "stages": measured,
                     "reciprocal_reference": reciprocal_analysis(cpu, gpu),
                     "rounding_boundaries": midpoint_analysis(cpu, gpu)})
    assert rows[1]["value17_exact"], "Warm normalization input is no longer byte-exact"
    result = {"schema_version": 1, "script_sha256": sha(Path(__file__)),
              "sources": {"replay": {"path": str(replay_path.relative_to(ROOT)), "sha256": sha(replay_path)},
                          "residual_summary": {"path": str(summary_path.relative_to(ROOT)),
                                               "sha256": sha(summary_path)},
                          "protocol": {"path": str(PROTOCOL.relative_to(ROOT)), "sha256": sha(PROTOCOL)},
                          "group_replay": {"path": str(GROUP_REPLAY.relative_to(ROOT)),
                                           "sha256": sha(GROUP_REPLAY)}},
              "archive_sha256": {name: meta["sha256"] for name, meta in replay["files"].items()},
              "controls_passed": True, "ordered_taps": ordered, "chunks": rows,
              "warm_first_differing_operation": rows[1]["first_differing_operation"],
              "scope": "Same saved CPU inputs and states, candidate group 0 only. CPU/GPU differences do not identify which sqrt result is correctly rounded; no full-model or speed claim."}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({"warm_first_differing_operation": result["warm_first_differing_operation"],
                      "warm_direct_inputs_exact": rows[1]["first_difference_direct_inputs_exact"],
                      "output": str(output_path.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
