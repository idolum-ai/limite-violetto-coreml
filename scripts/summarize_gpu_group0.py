"""Independently audit a completed group-0 CPU/GPU-policy diagnostic.

Reads saved JSON and NPZ evidence only. Does not load a Core ML model or infer
which device actually executed a CPU_AND_GPU prediction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import stat

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_SHA = "bd608693337d5dc092b544ff7828bd63fc4c62557e6cee1a3dc156768c585682"
PROTOCOL_SHA = "bf9c063364bc7c42e65a8d994bcd5a7ac8042cde37fcd3555dbda19e12c99731"
FIVE_E_SHA = "9d9e0656b224975d27551ca489f3448c69d62bc7401c4cef48d232f7b50a6606"
GENERATION_SHA = "1ee6553885123a57b8fb40b89c292eea201e148e6f3bd4d9b4cba1c057f9f3eb"
SOURCE_ID = "5a3e4c7a571e4fc3532377d9b9bb0da023f499c10709216fee7e1001bf4ad2c6"
COMPILED_ID = "a838dd36e8f87c6b9fc0dd7865d0912651c6ab3399fa299d10ff12d788bcb031"
OUTPUTS = ("y", "tap23_out")
STATES = tuple(f"inner_{kind}_cache_{layer}" for layer in range(4) for kind in ("k", "v"))
INPUTS = ("x", "ve", "positions", "tap0", "tap12", "tap23", "norm_divisor")
CASES = ("cold-prefill-3", "cold-prefill-32", "warm-32", "warm-128", "warm-512")
EXPECTED_GUARD = {"installed": True, "plain_calls": 2, "state_calls": 18, "gil_releases": 20}


class AuditError(ValueError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise AuditError(message)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"Cannot read JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def regular_file(path: Path) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not stat.S_ISLNK(mode), f"Not a real regular file: {path}")


def read_npz(path: Path, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    regular_file(path)
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == set(names), f"Unexpected arrays: {path}")
        return {name: archive[name].copy() for name in names}


def scalar(value):
    number = float(value)
    return number if math.isfinite(number) else None


def compare(expected: dict, actual: dict, names: tuple[str, ...]) -> dict:
    require(set(expected) == set(names) and set(actual) == set(names), "Comparison map names differ")
    rows = {}
    for name in names:
        a, b = expected[name], actual[name]
        shape_equal = a.shape == b.shape
        dtype_equal = a.dtype == b.dtype
        exact = bool(shape_equal and dtype_equal and np.array_equal(a, b))
        row = {"exact": exact, "shape_equal": shape_equal, "dtype_equal": dtype_equal,
               "expected_shape": list(a.shape), "actual_shape": list(b.shape),
               "expected_dtype": str(a.dtype), "actual_dtype": str(b.dtype)}
        if shape_equal:
            finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
            row["finite"] = finite
            if finite:
                delta = b.astype(np.float64) - a.astype(np.float64)
                row["max_abs"] = float(np.max(np.abs(delta))) if delta.size else 0.0
                row["rmse"] = float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0
                reference_rms = float(np.sqrt(np.mean(a.astype(np.float64) ** 2))) if a.size else 0.0
                row["relative_rmse"] = (row["rmse"] / reference_rms if reference_rms
                                        else (0.0 if row["rmse"] == 0 else None))
            else:
                row.update(max_abs=None, rmse=None, relative_rmse=None)
            if not exact:
                locations = np.argwhere(a != b)
                if len(locations):
                    index = tuple(int(index) for index in locations[0])
                    row["first_difference"] = {"index": list(index),
                                               "expected": scalar(a[index]),
                                               "actual": scalar(b[index])}
                else:
                    row["first_difference"] = None
        rows[name] = row
    return {"exact": all(row["exact"] for row in rows.values()), "arrays": rows}


def active_state_metrics(expected: dict, actual: dict, positions: np.ndarray) -> dict:
    """Separate the updated KV slots from the unchanged 2048-slot cache."""
    require(positions.ndim == 1 and positions.dtype == np.int32 and len(positions) > 0 and
            len(set(int(x) for x in positions)) == len(positions) and
            all(0 <= int(x) < 2048 for x in positions), "Invalid active cache positions")
    rows = {}
    for name in STATES:
        a, b = expected[name], actual[name]
        require(a.shape == b.shape and a.ndim == 4 and a.shape[1] == 2048,
                f"Invalid state shape for active-window diagnostic: {name}")
        active_a, active_b = a[:, positions], b[:, positions]
        def metrics(left, right):
            changed = int(np.count_nonzero(left != right))
            finite = bool(np.isfinite(left).all() and np.isfinite(right).all())
            result = {"changed_elements": changed, "total_elements": int(left.size),
                      "changed_fraction": changed / left.size, "finite": finite}
            if finite:
                diff = right.astype(np.float64) - left.astype(np.float64)
                result["absolute_rmse"] = float(np.sqrt(np.mean(diff * diff)))
                result["max_abs"] = float(np.max(np.abs(diff)))
            else:
                result["absolute_rmse"] = None
                result["max_abs"] = None
            return result
        rows[name] = {"full_cache": metrics(a, b), "active_positions": metrics(active_a, active_b)}
    return {"positions": [int(x) for x in positions], "states": rows}


def inventory(attempt: Path, recorded: list) -> dict[str, str]:
    require(isinstance(recorded, list), "Missing evidence inventory")
    rows = {}
    for entry in recorded:
        require(isinstance(entry, dict) and set(entry) == {"path", "size_bytes", "sha256"},
                "Malformed inventory entry")
        name = entry["path"]
        require(isinstance(name, str) and name and name not in rows and
                not Path(name).is_absolute() and ".." not in Path(name).parts,
                "Unsafe or duplicate inventory path")
        rows[name] = entry
    actual = {}
    for path in attempt.rglob("*"):
        mode = path.lstat().st_mode
        require(stat.S_ISDIR(mode) or stat.S_ISREG(mode), f"Nonregular evidence entry: {path}")
        require(not stat.S_ISLNK(mode), f"Symlink evidence entry: {path}")
        if stat.S_ISREG(mode) and path.name != "results.json":
            actual[str(path.relative_to(attempt))] = path
    require(set(actual) == set(rows), "Evidence inventory file set differs")
    for name, path in actual.items():
        entry = rows[name]
        require(path.stat().st_size == entry["size_bytes"] and sha(path) == entry["sha256"],
                f"Evidence inventory mismatch: {name}")
    return {name: entry["sha256"] for name, entry in rows.items()}


def audit(results_path: Path) -> dict:
    regular_file(results_path)
    attempt = results_path.parent
    raw = read_json(results_path)
    require(raw.get("status") == "completed" and raw.get("experiment_completed") is True,
            "Native probe did not complete")
    require(raw.get("script_sha256") == SCRIPT_SHA and raw.get("protocol_sha256") == PROTOCOL_SHA
            and raw.get("five_e_protocol_sha256") == FIVE_E_SHA,
            "Frozen script/protocol identity differs")
    require(raw.get("source_candidate_identity") == SOURCE_ID and
            raw.get("compiled_identity") == COMPILED_ID,
            "Selected source/compiled identity differs")
    selection = raw.get("artifact_selection", {})
    require(selection.get("group") == "0" and selection.get("variant_used") is False and
            raw.get("compute_units") == {"control": "CPU_ONLY", "diagnostic": "CPU_AND_GPU"},
            "Wrong graph or compute-policy selection")
    require(raw.get("loader_provenance") == raw.get("current_loader_provenance") and
            raw["loader_provenance"].get("source_candidate_identity") == SOURCE_ID and
            raw["loader_provenance"].get("compiled_aggregate_sha256") == COMPILED_ID,
            "Loader provenance differs")
    require(raw.get("runtime_provenance_before") == raw.get("runtime_provenance_after") and
            all(raw.get(field) is True for field in
                ("source_preserved", "compiled_preserved", "runtime_preserved",
                 "guard_counts_match_call_plan")), "Preservation gate differs")
    require(raw.get("prediction_guard_final") == EXPECTED_GUARD,
            "Prediction guard counts differ from 2 plain / 18 state / 20 releases")
    require(raw.get("execution_device_verified") is False and raw.get("performance_claim") is None,
            "Probe overclaims device placement or performance")
    require(raw.get("generation_fixture_sha256") == GENERATION_SHA and
            raw.get("stable_file_hashes_after") == {
                "script": SCRIPT_SHA, "protocol": PROTOCOL_SHA,
                "five_e_protocol": FIVE_E_SHA, "generation_fixture": GENERATION_SHA},
            "Stable-file provenance differs")
    pinned = {"source-snapshot.py": SCRIPT_SHA, "protocol-snapshot.md": PROTOCOL_SHA}
    measured = inventory(attempt, raw.get("evidence_inventory"))
    for name, digest in pinned.items():
        require(measured.get(name) == digest, f"Evidence snapshot changed: {name}")
    require(sha(ROOT / "scripts/probe_gpu_group0.py") == SCRIPT_SHA and
            sha(ROOT / "evidence/06-gpu/PROTOCOL.md") == PROTOCOL_SHA and
            sha(ROOT / "evidence/05e-product/protocol.json") == FIVE_E_SHA and
            sha(ROOT / "evidence/04-coreml/generation-comparison.json") == GENERATION_SHA,
            "Current pinned script or protocol differs")

    calls = raw.get("calls")
    require(isinstance(calls, list) and len(calls) == 18, "Expected 18 group call rows")
    call_map = {}
    for row in calls:
        require(isinstance(row, dict), "Malformed group call row")
        key = (row.get("case"), row.get("arm"), row.get("repeat"))
        require(key not in call_map, f"Duplicate group call: {key}")
        call_map[key] = row
    expected = {(case, "CPU_AND_GPU", repeat) for case in CASES for repeat in range(2)}
    expected |= {(case, "CPU_ONLY", repeat) for case in CASES[2:] for repeat in range(2)}
    expected |= {(case, "CPU_ONLY", 0) for case in CASES[:2]}
    require(set(call_map) == expected, "Group call matrix differs from 8 CPU / 10 GPU")
    require(raw.get("cpu_state_call_count") == 8 and raw.get("gpu_state_call_count") ==
            raw.get("gpu_replay_count"), "Recorded state call counts differ")

    reference = {}
    positions_by_case = {}
    cpu_audits = []
    for case in CASES:
        if case.startswith("cold"):
            ref_folder = attempt / case / "baseline"
        else:
            ref_folder = attempt / case / "sealed-reference"
        reference[case] = {"outputs": read_npz(ref_folder / "outputs.npz", OUTPUTS),
                           "states": read_npz(ref_folder / "state-after.npz", STATES)}
        input_folder = attempt / case if case.startswith("cold") else ref_folder
        positions_by_case[case] = read_npz(input_folder / "inputs.npz", INPUTS)["positions"]
        cpu_repeats = (0,) if case.startswith("cold") else (0, 1)
        for repeat in cpu_repeats:
            row = call_map[(case, "CPU_ONLY", repeat)]
            folder = attempt / case / ("baseline" if case.startswith("cold") else f"cpu-repeat-{repeat}")
            outputs = compare(reference[case]["outputs"], read_npz(folder / "outputs.npz", OUTPUTS), OUTPUTS)
            states = compare(reference[case]["states"], read_npz(folder / "state-after.npz", STATES), STATES)
            require(outputs["exact"] and states["exact"] and row.get("strict_exact") is True,
                    f"CPU reference/control differs: {case} repeat {repeat}")
            cpu_audits.append({"case": case, "repeat": repeat, "exact": True})

    gpu_audits = []
    earliest = None
    first_changed_state = None
    successful = 0
    for case in CASES:
        for repeat in range(2):
            row = call_map[(case, "CPU_AND_GPU", repeat)]
            folder = attempt / case / f"gpu-repeat-{repeat}"
            if "error" in row:
                require(row.get("strict_exact") is False, f"Errored GPU row claims exactness: {case}/{repeat}")
                gpu_audits.append({"case": case, "repeat": repeat, "operational": "error",
                                   "error": row["error"], "exact": False})
                if earliest is None:
                    earliest = {"case": case, "repeat": repeat, "kind": "operational_error"}
                continue
            outputs = compare(reference[case]["outputs"], read_npz(folder / "outputs.npz", OUTPUTS), OUTPUTS)
            states = compare(reference[case]["states"], read_npz(folder / "state-after.npz", STATES), STATES)
            successful += 1
            exact = outputs["exact"] and states["exact"]
            require(row.get("strict_exact") is exact,
                    f"Recorded GPU exactness differs from saved arrays: {case}/{repeat}")
            require(row.get("state_restoration", {}).get("exact") is True,
                    f"GPU state restoration was not exact: {case}/{repeat}")
            for field, recalculated in (("outputs_vs_cpu", outputs), ("states_vs_cpu", states)) if case.startswith("cold") else (("outputs_vs_fixture", outputs), ("states_vs_fixture", states)):
                require(row.get(field, {}).get("exact") is recalculated["exact"],
                        f"Recorded per-map exactness differs: {case}/{repeat}/{field}")
            changed_outputs = [name for name in OUTPUTS if not outputs["arrays"][name]["exact"]]
            changed_states = [name for name in STATES if not states["arrays"][name]["exact"]]
            if earliest is None and (changed_outputs or changed_states):
                kind, name, item = (("output", changed_outputs[0], outputs) if changed_outputs else
                                    ("state", changed_states[0], states))
                earliest = {"case": case, "repeat": repeat, "kind": kind, "name": name,
                            "first_difference": item["arrays"][name].get("first_difference")}
            if first_changed_state is None and changed_states:
                name = changed_states[0]
                first_changed_state = {"case": case, "repeat": repeat, "name": name,
                                       "first_difference": states["arrays"][name].get("first_difference")}
            gpu_audits.append({"case": case, "repeat": repeat, "operational": "completed",
                               "exact": exact, "outputs": outputs, "states": states,
                               "active_state_diagnostics": active_state_metrics(
                                   reference[case]["states"],
                                   read_npz(folder / "state-after.npz", STATES),
                                   positions_by_case[case]),
                               "changed_outputs": changed_outputs, "changed_states": changed_states})
    require(successful == raw.get("gpu_replay_count"), "Recorded successful GPU replay count differs")
    strict = successful == 10 and all(row["exact"] for row in gpu_audits)
    require(raw.get("strict_same_graph_device_parity_passed") is strict and
            raw.get("strict_fidelity_status") == ("exact" if strict else "not_exact") and
            raw.get("operational_status") == ("calls_completed" if successful == 10 else "gpu_call_errors_recorded"),
            "Recorded operational/fidelity decision differs")
    expected_decision = "compute-policy exact parity" if strict else "compute-policy exactness no-go"
    require(expected_decision in raw.get("decision", ""), "Decision text differs from raw arrays")

    repeats = {}
    for case in CASES:
        first, second = gpu_audits[2 * CASES.index(case):2 * CASES.index(case) + 2]
        if first["operational"] != "completed" or second["operational"] != "completed":
            repeats[case] = {"available": False, "reason": "GPU call error"}
            continue
        folder0 = attempt / case / "gpu-repeat-0"
        folder1 = attempt / case / "gpu-repeat-1"
        outputs = compare(read_npz(folder0 / "outputs.npz", OUTPUTS),
                          read_npz(folder1 / "outputs.npz", OUTPUTS), OUTPUTS)
        states = compare(read_npz(folder0 / "state-after.npz", STATES),
                         read_npz(folder1 / "state-after.npz", STATES), STATES)
        repeats[case] = {"available": True, "outputs_exact": outputs["exact"],
                         "states_exact": states["exact"]}

    finite_abs = [array["max_abs"] for row in gpu_audits if row.get("operational") == "completed"
                  for group in ("outputs", "states") for array in row[group]["arrays"].values()
                  if array.get("max_abs") is not None]
    return {"status": "completed", "audit_passed": True,
            "summarizer_sha256": sha(Path(__file__)),
            "native_result_sha256": sha(results_path), "native_status": raw["status"],
            "operational_status": raw["operational_status"],
            "strict_fidelity_status": raw["strict_fidelity_status"],
            "strict_compute_policy_value_parity": strict,
            "actual_execution_device_verified": False, "performance_claim": None,
            "source_candidate_identity": SOURCE_ID, "compiled_identity": COMPILED_ID,
            "guard": EXPECTED_GUARD, "evidence_files_verified": len(measured),
            "group_calls": {"cpu": 8, "gpu_policy": 10, "gpu_completed": successful},
            "cpu_controls": cpu_audits, "gpu_replays": gpu_audits,
            "gpu_repeat_agreement": repeats, "earliest_difference_or_error": earliest,
            "first_changed_state": first_changed_state,
            "largest_finite_absolute_difference": max(finite_abs, default=None),
            "state_diagnostic_note": "Full 2048-slot cache RMSE averages mostly unchanged slots; active_positions metrics isolate slots written by this call. Neither diagnostic replaces exact value parity.",
            "compute_plan_status": (raw.get("compute_plan") or {}).get("status"),
            "interpretation": "Same compiled group under CPU_ONLY and CPU_AND_GPU policies; actual execution device unverified. No latency or full-model claim."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "evidence/06-gpu/summary.json")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output exists; refusing to overwrite: {args.output}")
    summary = audit(args.results.expanduser().resolve(strict=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"audit_passed": summary["audit_passed"],
                      "strict_compute_policy_value_parity": summary["strict_compute_policy_value_parity"],
                      "operational_status": summary["operational_status"],
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
