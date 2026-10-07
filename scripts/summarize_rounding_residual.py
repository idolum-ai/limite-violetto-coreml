"""Independently audit saved group-0 residual tap replays without inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GROUP_REPLAY = "evidence/06d-gpu-rounding/group0-replay-1/results.json"
ENDPOINTS = ("y", "tap23_out")


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return result


def load(meta: dict) -> dict[str, np.ndarray]:
    path = ROOT / meta["path"]
    if sha(path) != meta["sha256"]:
        raise ValueError(f"Archive changed: {path}")
    with np.load(path) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    if set(arrays) != set(meta["shapes"]) or set(arrays) != set(meta["dtypes"]):
        raise ValueError(f"Archive schema changed: {path}")
    for name, array in arrays.items():
        if list(array.shape) != meta["shapes"][name] or str(array.dtype) != meta["dtypes"][name]:
            raise ValueError(f"Array schema changed: {path}:{name}")
        if not np.isfinite(array).all():
            raise ValueError(f"Nonfinite array: {path}:{name}")
    return arrays


def metric(actual: np.ndarray, expected: np.ndarray) -> dict:
    if actual.shape != expected.shape or actual.dtype != expected.dtype or actual.dtype != np.float32:
        raise ValueError("FP32 array schema mismatch")
    a = np.ascontiguousarray(actual).view(np.uint32)
    b = np.ascontiguousarray(expected).view(np.uint32)
    changed = a != b
    numeric_changed = actual != expected
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    rmse = float(np.sqrt(np.mean(delta * delta)))
    expected_rms = float(np.sqrt(np.mean(expected.astype(np.float64) ** 2)))
    first = None
    if np.any(changed):
        index = tuple(int(i) for i in np.argwhere(changed)[0])
        first = {"index": list(index), "actual": float(actual[index]),
                 "expected": float(expected[index]),
                 "actual_bits": f"0x{int(a[index]):08x}",
                 "expected_bits": f"0x{int(b[index]):08x}"}
    rows = np.flatnonzero(np.any(changed, axis=tuple(i for i in range(changed.ndim) if i != 1)))
    return {"shape": list(actual.shape), "dtype": str(actual.dtype),
            "bitwise_exact": not bool(np.any(changed)),
            "bitwise_error_count": int(np.count_nonzero(changed)),
            "numeric_error_count": int(np.count_nonzero(numeric_changed)),
            "max_abs": float(np.max(np.abs(delta))),
            "mean_abs": float(np.mean(np.abs(delta))),
            "rmse": rmse, "relative_rmse": rmse / expected_rms if expected_rms else None,
            "query_rows_with_differences": [int(i) for i in rows],
            "first_difference": first}


def reported_matches(measured: dict, reported: dict) -> None:
    assert measured["shape"] == reported["actual_shape"] == reported["expected_shape"]
    assert measured["dtype"] == reported["actual_dtype"] == reported["expected_dtype"]
    assert measured["bitwise_exact"] == reported["bitwise_exact"] == reported["exact"]
    for key in ("max_abs", "rmse", "relative_rmse"):
        expected = reported[key]
        actual = measured[key]
        if expected is None or actual is None:
            assert expected is actual
        else:
            assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), (key, actual, expected)
    first = reported["first_difference"]
    ours = measured["first_difference"]
    if first is None:
        assert ours is None
    else:
        assert ours is not None and ours["index"] == first["index"]
        assert ours["actual"] == first["actual"] and ours["expected"] == first["expected"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    replay_path = (ROOT / args.replay).resolve()
    output_path = (ROOT / args.output).resolve()
    if not replay_path.is_relative_to(ROOT) or not output_path.is_relative_to(ROOT):
        raise ValueError("Replay and output must remain inside the repository")
    if output_path.exists():
        raise FileExistsError(f"Fresh summary output required: {output_path}")
    replay = read(replay_path)
    assert replay["status"] == "completed" and replay["passed"] and replay["instrumentation_valid"]
    assert replay["prediction_guard"] == {"installed": True, "plain_calls": 0,
                                          "state_calls": 8, "gil_releases": 8}
    assert len(replay["controls"]) == 4 and len(replay["calls"]) == 8
    assert len(replay["tap_comparisons"]) == 2
    taps = replay["taps"]
    assert taps and len(taps) == len(set(taps))
    for relative, expected in replay["pins"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"Pinned source changed: {relative}")
    assert GROUP_REPLAY in replay["pins"]
    group = read(ROOT / GROUP_REPLAY)
    assert group["status"] == "completed" and group["diagnostic_completed"]
    files = replay["files"]
    assert len(files) == 16
    archives = {name: load(meta) for name, meta in files.items()}
    group_archives = {name: load(meta) for name, meta in group["files"].items()
                      if name.endswith("/outputs") or name.endswith("/state-after")}
    controls = {(row["chunk"], row["policy"]): row for row in replay["controls"]}
    assert set(controls) == {(chunk, policy) for chunk in (0, 1) for policy in ("CPU", "GPU")}
    calls = {(row["chunk"], row["policy"], row["arm"]): row for row in replay["calls"]}
    assert set(calls) == {(chunk, policy, arm) for chunk in (0, 1)
                         for policy in ("CPU", "GPU") for arm in ("original", "tapped")}
    reported = {row["chunk"]: row["comparisons"] for row in replay["tap_comparisons"]}
    assert set(reported) == {0, 1}
    summary = {"schema_version": 1, "script_sha256": sha(Path(__file__)),
               "replay": {"path": str(replay_path.relative_to(ROOT)), "sha256": sha(replay_path)},
               "pinned_sources": replay["pins"],
               "archive_sha256": {name: meta["sha256"] for name, meta in files.items()},
               "taps": taps, "controls": [], "chunks": [],
               "scope": "Saved same-input/same-state group-0 taps and endpoints only; no native execution, full-model conclusion, or speed measurement."}
    for chunk in (0, 1):
        control_chunk = []
        for policy in ("CPU", "GPU"):
            row = controls[(chunk, policy)]
            assert row["passed"]
            original_out = archives[f"chunk{chunk}/{policy}-original/outputs"]
            original_state = archives[f"chunk{chunk}/{policy}-original/state-after"]
            tapped_out = archives[f"chunk{chunk}/{policy}-tapped/outputs"]
            tapped_state = archives[f"chunk{chunk}/{policy}-tapped/state-after"]
            saved_arm = "original-CPU" if policy == "CPU" else "candidate-GPU"
            saved_out = group_archives[f"chunk{chunk}/{saved_arm}/outputs"]
            saved_state = group_archives[f"chunk{chunk}/{saved_arm}/state-after"]
            assert set(original_out) == set(saved_out) == set(ENDPOINTS)
            assert set(tapped_out) == set(ENDPOINTS) | set(taps)
            assert set(original_state) == set(tapped_state) == set(saved_state)
            checks = {
                "original_outputs_vs_saved": {name: metric(original_out[name], saved_out[name])
                                              for name in ENDPOINTS},
                "original_states_vs_saved": {name: metric(original_state[name], saved_state[name])
                                             for name in original_state},
                "tapped_endpoints_vs_original": {name: metric(tapped_out[name], original_out[name])
                                                 for name in ENDPOINTS},
                "tapped_states_vs_original": {name: metric(tapped_state[name], original_state[name])
                                              for name in original_state},
            }
            for category, entries in checks.items():
                assert set(entries) == set(row[category])
                for name, result in entries.items():
                    reported_matches(result, row[category][name])
                    assert result["bitwise_exact"], (chunk, policy, category, name)
            for arm in ("original", "tapped"):
                call = calls[(chunk, policy, arm)]
                prefix = f"chunk{chunk}/{policy}-{arm}"
                assert call["restoration_exact"]
                assert call["outputs"]["sha256"] == files[f"{prefix}/outputs"]["sha256"]
                assert call["state_after"]["sha256"] == files[f"{prefix}/state-after"]["sha256"]
            control_chunk.append({"policy": policy, "passed": True,
                                  "original_output_sha256": files[f"chunk{chunk}/{policy}-original/outputs"]["sha256"],
                                  "original_state_sha256": files[f"chunk{chunk}/{policy}-original/state-after"]["sha256"],
                                  "tapped_output_sha256": files[f"chunk{chunk}/{policy}-tapped/outputs"]["sha256"],
                                  "tapped_state_sha256": files[f"chunk{chunk}/{policy}-tapped/state-after"]["sha256"]})
        cpu = archives[f"chunk{chunk}/CPU-tapped/outputs"]
        gpu = archives[f"chunk{chunk}/GPU-tapped/outputs"]
        assert set(reported[chunk]) == set(taps)
        comparisons = {}
        for name in taps:
            result = metric(gpu[name], cpu[name])
            reported_matches(result, reported[chunk][name])
            comparisons[name] = result
        summary["controls"].extend({"chunk": chunk, **entry} for entry in control_chunk)
        summary["chunks"].append({"chunk": chunk, "query_length": int(cpu[taps[0]].shape[1]),
                                  "tap_comparisons": comparisons,
                                  "first_changed_tap": next((name for name in taps if not comparisons[name]["bitwise_exact"]), None)})
    assert len(summary["controls"]) == 4 and all(row["passed"] for row in summary["controls"])
    summary["controls_passed"] = True
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({"controls_passed": True,
                      "first_changed_tap": [c["first_changed_tap"] for c in summary["chunks"]],
                      "output": str(output_path.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
