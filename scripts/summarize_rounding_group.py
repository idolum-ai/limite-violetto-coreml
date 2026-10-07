"""Independent array audit of the isolated v3 group-0 replay.

Uses the original CPU-input/CPU-state/GPU-policy replay as the matched baseline.
This is a fidelity diagnostic for group 0, not full-model or speed evidence.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPLAY = ROOT / "evidence/06d-gpu-rounding/group0-replay-1/results.json"
PROMPT = ROOT / "evidence/06c-gpu-localization/prompt0-1/results.json"
BUILD = ROOT / "evidence/06d-gpu-rounding/group0-1/results.json"
OPERATOR = ROOT / "evidence/06d-gpu-rounding/round-replacement-1/results.json"
OUTPUT = ROOT / "evidence/06d-gpu-rounding/group-summary.json"
STATE_NAMES = tuple(f"inner_{kind}_cache_{index}" for kind in ("k", "v") for index in range(4))
OUTPUT_NAMES = ("tap23_out", "y")


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected object: {path}")
    return value


def archive(meta: dict) -> dict[str, np.ndarray]:
    path = ROOT / meta["path"]
    if sha(path) != meta["sha256"]:
        raise ValueError(f"Archive hash changed: {path}")
    with np.load(path) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    assert set(arrays) == set(meta["dtypes"]) == set(meta["shapes"])
    for name, value in arrays.items():
        assert str(value.dtype) == meta["dtypes"][name]
        assert list(value.shape) == meta["shapes"][name]
        assert np.isfinite(value).all()
    return arrays


def bits_equal(a: np.ndarray, b: np.ndarray) -> bool:
    assert a.dtype == b.dtype == np.float32 and a.shape == b.shape
    return bool(np.array_equal(a.view(np.uint32), b.view(np.uint32)))


def metric(a: np.ndarray, ref: np.ndarray, offset: int = 0) -> dict:
    assert a.dtype == ref.dtype == np.float32 and a.shape == ref.shape
    assert np.isfinite(a).all() and np.isfinite(ref).all()
    actual_bits = np.ascontiguousarray(a).view(np.uint32)
    reference_bits = np.ascontiguousarray(ref).view(np.uint32)
    bit_differences = actual_bits != reference_bits
    numeric_differences = a != ref
    delta = a.astype(np.float64) - ref.astype(np.float64)
    abs_delta = np.abs(delta)
    rmse = float(np.sqrt(np.mean(delta * delta)))
    ref_rms = float(np.sqrt(np.mean(ref.astype(np.float64) ** 2)))
    first = None
    if np.any(bit_differences):
        local = tuple(int(i) for i in np.argwhere(bit_differences)[0])
        index = list(local)
        if a.ndim == 4:
            index[1] += offset
        first = {"index": index, "actual": float(a[local]), "reference": float(ref[local]),
                 "actual_bits": f"0x{int(actual_bits[local]):08x}",
                 "reference_bits": f"0x{int(reference_bits[local]):08x}"}
    return {"elements": int(a.size), "bitwise_error_count": int(np.count_nonzero(bit_differences)),
            "numeric_error_count": int(np.count_nonzero(numeric_differences)),
            "mean_abs": float(np.mean(abs_delta)), "max_abs": float(np.max(abs_delta)),
            "rmse": rmse, "relative_rmse": rmse / ref_rms if ref_rms else None,
            "first_difference": first}


def comparison(baseline: np.ndarray, candidate: np.ndarray, cpu: np.ndarray,
               offset: int = 0) -> dict:
    old = metric(baseline, cpu, offset)
    new = metric(candidate, cpu, offset)
    return {"original_gpu_vs_original_cpu": old,
            "candidate_gpu_vs_original_cpu": new,
            "candidate_vs_original_gpu": metric(candidate, baseline, offset),
            "rmse_reduction_percent": 100.0 * (1.0 - new["rmse"] / old["rmse"])
            if old["rmse"] else None,
            "error_count_reduction": old["bitwise_error_count"] - new["bitwise_error_count"]}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite independent summary: {OUTPUT}")
    replay, prompt, build, operator = map(read, (REPLAY, PROMPT, BUILD, OPERATOR))
    assert replay["status"] == "completed" and replay["diagnostic_completed"]
    assert prompt["status"] == "completed" and prompt["passed"]
    assert build["status"] == "completed" and build["passed"]
    assert operator["status"] == "completed" and operator["bounded_replacement_gate_passed"]
    assert not operator["strict_full_q_torch_passed"]
    assert replay["build_sha256"] == sha(BUILD) and replay["operator_result_sha256"] == sha(OPERATOR)
    assert replay["original_source_identity"] == prompt["source_identity"] == build["source_candidate_identity"]
    assert replay["original_compiled_identity"] == prompt["compiled_identity"]
    assert replay["prediction_guard"] == {"installed": True, "plain_calls": 0,
                                            "state_calls": 6, "gil_releases": 6}
    assert len(replay["calls"]) == 6
    calls = {(row["chunk"], row["arm"], row["policy"]): row for row in replay["calls"]}
    assert set(calls) == {(chunk, arm, policy) for chunk in (0, 1)
                         for arm, policy in (("original", "CPU"), ("candidate", "CPU"), ("candidate", "GPU"))}
    summary = {"schema_version": 1, "script_sha256": sha(Path(__file__)),
               "sources": {"group_replay": {"path": str(REPLAY.relative_to(ROOT)), "sha256": sha(REPLAY)},
                           "matched_baseline": {"path": str(PROMPT.relative_to(ROOT)), "sha256": sha(PROMPT)},
                           "group_build": {"path": str(BUILD.relative_to(ROOT)), "sha256": sha(BUILD)},
                           "round_operator": {"path": str(OPERATOR.relative_to(ROOT)), "sha256": sha(OPERATOR)}},
               "original_source_identity": replay["original_source_identity"],
               "original_compiled_identity": replay["original_compiled_identity"],
               "fixture_sha256": replay["fixture_sha256"],
               "operator_limits": {"original_cpu_round_control_passed": operator["original_cpu_round_control_passed"],
                                   "round_replacement_gate_passed": operator["round_replacement_gate_passed"],
                                   "native_cpu_full_q_preservation_passed": operator["native_cpu_full_q_preservation_passed"],
                                   "strict_full_q_torch_passed": operator["strict_full_q_torch_passed"],
                                   "baseline_cpu_negative_zero_changes": operator["baseline_cpu_negative_zero_changes"]},
               "six_calls": [], "chunks": [],
               "scope": "Same saved CPU inputs and CPU-before-state, original GPU-policy replay vs original CPU and candidate GPU-policy. Group 0 only; GPU placement and speed unverified."}
    cpu_control_passed = True
    for chunk in (0, 1):
        group = next(g for g in prompt["groups"] if g["chunk"] == chunk and g["group"] == 0)
        start, count = int(group["start"]), int(group["count"])
        assert (start, count) == ((0, 32) if chunk == 0 else (32, 12))
        replay_gpu = next(r for r in group["replays"] if r["label"] == "ICPU-SCPU-PGPU")
        assert replay_gpu["input_trajectory"] == replay_gpu["state_trajectory"] == "CPU"
        assert replay_gpu["compute_policy"] == "GPU" and replay_gpu["restoration_exact"]
        prefix = f"chunk{chunk}"
        prompt_prefix = f"chunk{chunk}/group0"
        same_input = replay["files"][f"{prefix}/inputs"]["sha256"] == \
            prompt["files"][f"{prompt_prefix}/own-CPU/inputs"]["sha256"]
        same_before = replay["files"][f"{prefix}/before"]["sha256"] == \
            prompt["files"][f"{prompt_prefix}/own-CPU/state-before"]["sha256"]
        assert same_input and same_before
        input_arrays = archive(replay["files"][f"{prefix}/inputs"])
        before_arrays = archive(replay["files"][f"{prefix}/before"])
        assert input_arrays and set(before_arrays) == set(STATE_NAMES)
        cpu_out = archive(replay["files"][f"{prefix}/original-CPU/outputs"])
        cpu_state = archive(replay["files"][f"{prefix}/original-CPU/state-after"])
        candidate_cpu_out = archive(replay["files"][f"{prefix}/candidate-CPU/outputs"])
        candidate_cpu_state = archive(replay["files"][f"{prefix}/candidate-CPU/state-after"])
        candidate_gpu_out = archive(replay["files"][f"{prefix}/candidate-GPU/outputs"])
        candidate_gpu_state = archive(replay["files"][f"{prefix}/candidate-GPU/state-after"])
        baseline_gpu_out = archive(replay_gpu["outputs"])
        baseline_gpu_state = archive(replay_gpu["state-after"])
        prompt_cpu_out = archive(prompt["files"][f"{prompt_prefix}/own-CPU/outputs"])
        prompt_cpu_state = archive(prompt["files"][f"{prompt_prefix}/own-CPU/state-after"])
        assert set(cpu_out) == set(candidate_cpu_out) == set(candidate_gpu_out) == \
            set(baseline_gpu_out) == set(prompt_cpu_out) == set(OUTPUT_NAMES)
        assert set(cpu_state) == set(candidate_cpu_state) == set(candidate_gpu_state) == \
            set(baseline_gpu_state) == set(prompt_cpu_state) == set(STATE_NAMES)
        cpu_outputs_exact = all(bits_equal(cpu_out[n], candidate_cpu_out[n]) and
                                bits_equal(cpu_out[n], prompt_cpu_out[n]) for n in OUTPUT_NAMES)
        cpu_states_exact = all(bits_equal(cpu_state[n], candidate_cpu_state[n]) and
                               bits_equal(cpu_state[n], prompt_cpu_state[n]) for n in STATE_NAMES)
        cpu_control_passed &= cpu_outputs_exact and cpu_states_exact
        assert cpu_outputs_exact and cpu_states_exact
        for arm, policy in (("original", "CPU"), ("candidate", "CPU"), ("candidate", "GPU")):
            call = calls[(chunk, arm, policy)]
            key = f"{prefix}/{arm}-{policy}"
            assert call["restoration_exact"] and call["outputs"]["sha256"] == \
                replay["files"][f"{key}/outputs"]["sha256"]
            assert call["state_after"]["sha256"] == replay["files"][f"{key}/state-after"]["sha256"]
            summary["six_calls"].append({"chunk": chunk, "arm": arm, "policy": policy,
                                         "output_archive_sha256": call["outputs"]["sha256"],
                                         "state_archive_sha256": call["state_after"]["sha256"],
                                         "restoration_exact": call["restoration_exact"]})
        outputs = {name: comparison(baseline_gpu_out[name], candidate_gpu_out[name], cpu_out[name])
                   for name in OUTPUT_NAMES}
        caches = {}
        for name in STATE_NAMES:
            a, b, c = baseline_gpu_state[name], candidate_gpu_state[name], cpu_state[name]
            active = np.s_[:, start:start + count, :, :]
            inactive_exact = bool(bits_equal(b[:, :start], c[:, :start]) and
                                  bits_equal(b[:, start + count:], c[:, start + count:]))
            caches[name] = {"active": comparison(a[active], b[active], c[active], start),
                            "full": comparison(a, b, c),
                            "candidate_inactive_exact": inactive_exact}
        summary["chunks"].append({"chunk": chunk, "start": start, "count": count,
                                  "input_archive_sha256": replay["files"][f"{prefix}/inputs"]["sha256"],
                                  "before_archive_sha256": replay["files"][f"{prefix}/before"]["sha256"],
                                  "original_cpu_output_sha256": replay["files"][f"{prefix}/original-CPU/outputs"]["sha256"],
                                  "original_cpu_state_sha256": replay["files"][f"{prefix}/original-CPU/state-after"]["sha256"],
                                  "matched_original_gpu_output_sha256": replay_gpu["outputs"]["sha256"],
                                  "matched_original_gpu_state_sha256": replay_gpu["state-after"]["sha256"],
                                  "cpu_output_and_state_preserved": cpu_outputs_exact and cpu_states_exact,
                                  "outputs": outputs, "caches": caches,
                                  "candidate_first_changed_active_state": next(
                                      (name for name in STATE_NAMES if caches[name]["active"]
                                       ["candidate_gpu_vs_original_cpu"]["bitwise_error_count"]), None)})
    summary["integrity_passed"] = bool(cpu_control_passed)
    summary["gpu_policy_exact_gate_passed"] = all(
        item["candidate_gpu_vs_original_cpu"]["bitwise_error_count"] == 0
        for chunk in summary["chunks"] for item in
        list(chunk["outputs"].values()) + [v["full"] for v in chunk["caches"].values()])
    assert summary["integrity_passed"] == replay["cpu_preservation_passed"]
    assert summary["gpu_policy_exact_gate_passed"] == replay["gpu_exact_gate_passed"]
    OUTPUT.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(json.dumps({"integrity_passed": summary["integrity_passed"],
                      "gpu_policy_exact_gate_passed": summary["gpu_policy_exact_gate_passed"],
                      "y_relative_rmse": [(c["outputs"]["y"]["original_gpu_vs_original_cpu"]["relative_rmse"],
                                           c["outputs"]["y"]["candidate_gpu_vs_original_cpu"]["relative_rmse"])
                                          for c in summary["chunks"]]}, indent=2))


if __name__ == "__main__":
    main()
