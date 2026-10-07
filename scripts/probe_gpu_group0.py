"""Bounded same-graph CPU_ONLY versus CPU_AND_GPU group-0 parity probe.

The script is intentionally opt-in and uses only the original sealed compiled
candidate. It makes no latency or full-model qualification claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EXPECTED_CANDIDATE_ID = "5a3e4c7a571e4fc3532377d9b9bb0da023f499c10709216fee7e1001bf4ad2c6"
EXPECTED_COMPILED_ID = "a838dd36e8f87c6b9fc0dd7865d0912651c6ab3399fa299d10ff12d788bcb031"
FIVE_E_PROTOCOL = ROOT / "evidence/05e-product/protocol.json"
FIVE_E_PROTOCOL_SHA = "9d9e0656b224975d27551ca489f3448c69d62bc7401c4cef48d232f7b50a6606"
SOURCE_ROOT = ROOT / "artifacts/04d-cpu/full-candidate-6"
SOURCE_MANIFEST = ROOT / "evidence/04d-cpu/full-candidate-6/package-identity.json"
COMPILED_ROOT = ROOT / "artifacts/05c-compiled/full-cache-1"
COMPILED_MANIFEST = ROOT / "evidence/05c-compiled/full-cache-1/compiled-identity.json"
MILESTONE = ROOT / "evidence/05c-compiled/milestone-rerun1.json"
EXPECTED_OUTPUTS = {"y", "tap23_out"}
EXPECTED_STATES = {f"inner_{kv}_cache_{i}" for i in range(4) for kv in ("k", "v")}
PREFIXES = (32, 128, 512)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_npz(path: Path, arrays: dict) -> None:
    import numpy as np
    np.savez(path, **arrays)


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def save_outputs(out: Path, label: str, outputs: dict, states: dict) -> None:
    folder = out / label
    folder.mkdir(parents=True, exist_ok=False)
    save_npz(folder / "outputs.npz", outputs)
    save_npz(folder / "state-after.npz", states)


def first_difference(expected, actual):
    import numpy as np
    where = np.argwhere(expected != actual)
    if not len(where):
        return None
    index = tuple(int(x) for x in where[0])
    def scalar(value):
        number = float(value)
        return number if np.isfinite(number) else None
    return {"index": list(index), "expected": scalar(expected[index]), "actual": scalar(actual[index])}


def compare_array(expected, actual) -> dict:
    import numpy as np
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    same_shape = expected.shape == actual.shape
    same_dtype = expected.dtype == actual.dtype
    exact = same_shape and same_dtype and np.array_equal(expected, actual)
    result = {"exact": bool(exact), "expected_shape": list(expected.shape),
              "actual_shape": list(actual.shape), "expected_dtype": str(expected.dtype),
              "actual_dtype": str(actual.dtype)}
    if not same_shape:
        result["shape_match"] = False
        return result
    result["shape_match"] = True
    finite_expected = bool(np.isfinite(expected).all())
    finite_actual = bool(np.isfinite(actual).all())
    result["finite_counts"] = {"expected": int(np.isfinite(expected).sum()),
                               "actual": int(np.isfinite(actual).sum()),
                               "total": int(expected.size)}
    result["finite"] = finite_expected and finite_actual
    if not result["finite"]:
        result.update(max_abs=None, rmse=None, relative_rmse=None, cosine_similarity=None,
                      first_difference=first_difference(expected, actual))
        return result
    a = actual.astype(np.float64, copy=False)
    b = expected.astype(np.float64, copy=False)
    delta = a - b
    result["max_abs"] = float(np.max(np.abs(delta))) if delta.size else 0.0
    rmse = float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0
    reference_rms = float(np.sqrt(np.mean(b * b))) if b.size else 0.0
    result["rmse"] = rmse
    result["relative_rmse"] = rmse / reference_rms if reference_rms else (0.0 if rmse == 0 else None)
    a_norm = float(np.linalg.norm(a.ravel()))
    b_norm = float(np.linalg.norm(b.ravel()))
    result["cosine_similarity"] = (float(np.dot(a.ravel(), b.ravel()) / (a_norm * b_norm))
                                    if a_norm and b_norm else (1.0 if exact else None))
    result["first_difference"] = first_difference(expected, actual) if not exact else None
    return result


def compare_maps(expected: dict, actual: dict, expected_keys: set[str]) -> dict:
    rows = {}
    if set(expected) != expected_keys:
        return {"exact": False, "error": "expected map has wrong keys",
                "expected_keys": sorted(expected_keys), "observed_expected_keys": sorted(expected)}
    if set(actual) != expected_keys:
        rows["_key_set"] = {"exact": False, "expected_keys": sorted(expected_keys),
                            "actual_keys": sorted(actual)}
    else:
        rows.update({name: compare_array(expected[name], actual[name]) for name in sorted(expected_keys)})
    rows["exact"] = set(actual) == expected_keys and all(
        row.get("exact") is True for key, row in rows.items() if key != "exact")
    diagnostic_rows = [row for key, row in rows.items() if key != "exact" and "relative_rmse" in row]
    rows["diagnostic_thresholds"] = {
        "relative_rmse_max_0_01": bool(diagnostic_rows) and all(
            row.get("relative_rmse") is not None and row["relative_rmse"] <= 0.01 for row in diagnostic_rows),
        "cosine_min_0_995": bool(diagnostic_rows) and all(
            row.get("cosine_similarity") is not None and row["cosine_similarity"] >= 0.995 for row in diagnostic_rows),
        "scope": "diagnostic only; strict bitwise exactness is the same-graph device-parity gate",
    }
    abs_values = [row["max_abs"] for row in diagnostic_rows if isinstance(row.get("max_abs"), (int, float))]
    rel_values = [row["relative_rmse"] for row in diagnostic_rows if isinstance(row.get("relative_rmse"), (int, float))]
    cos_values = [row["cosine_similarity"] for row in diagnostic_rows if isinstance(row.get("cosine_similarity"), (int, float))]
    rows["aggregate"] = {"max_abs": max(abs_values, default=None),
                          "max_relative_rmse": max(rel_values, default=None),
                          "min_cosine_similarity": min(cos_values, default=None)}
    return rows


def _array_files(record: dict, fixture_json: Path) -> dict:
    import numpy as np
    if record.get("status") != "completed" or record.get("passed") is not True:
        raise ValueError(f"Capture is not a completed passing record: {fixture_json}")
    if record.get("group") != "0" or record.get("prefix_length") not in PREFIXES:
        raise ValueError(f"Unexpected group/prefix capture: {fixture_json}")
    if not isinstance(record.get("replays"), list) or len(record["replays"]) != 2 or not all(
        isinstance(item, dict) and item.get("outputs_exact") is True and
        item.get("all_eight_states_exact") is True for item in record["replays"]
    ):
        raise ValueError(f"Capture lacks two passing exact replays: {fixture_json}")
    arrays = {}
    for key in ("inputs", "state-before", "outputs", "state-after"):
        info = record.get("files", {}).get(key)
        path = fixture_json.parent / f"{key}.npz"
        if not isinstance(info, dict) or not path.is_file() or sha(path) != info.get("sha256"):
            raise ValueError(f"Capture NPZ missing or changed: {path}")
        with np.load(path) as stored:
            arrays[key] = {name: stored[name].copy() for name in stored.files}
    if set(arrays["outputs"]) != EXPECTED_OUTPUTS:
        raise ValueError(f"Unexpected output names in {fixture_json}")
    if set(arrays["state-before"]) != EXPECTED_STATES or set(arrays["state-after"]) != EXPECTED_STATES:
        raise ValueError(f"Capture does not contain all eight states: {fixture_json}")
    if set(arrays["state-before"]) != set(arrays["state-after"]):
        raise ValueError(f"State schemas disagree in {fixture_json}")
    if int(arrays["inputs"]["positions"][0]) != record["prefix_length"]:
        raise ValueError(f"Capture position differs from prefix metadata: {fixture_json}")
    return arrays


def load_fixtures(out: Path, expected_loader: dict, expected_runtime: dict, generation_stream: list[int]):
    import numpy as np
    protocol_path = FIVE_E_PROTOCOL
    if sha(protocol_path) != FIVE_E_PROTOCOL_SHA:
        raise ValueError("Pinned 05e protocol changed")
    protocol = json.loads(protocol_path.read_text())
    records = {}
    for item in protocol["fixtures"]:
        path = ROOT / item["results"]
        if sha(path) != item["sha256"]:
            raise ValueError(f"Pinned fixture result changed: {path}")
        record = json.loads(path.read_text())
        arrays = _array_files(record, path)
        prefix = record["prefix_length"]
        if prefix in records:
            raise ValueError(f"Duplicate fixture prefix: {prefix}")
        if record.get("loader_provenance") != expected_loader or record.get("runtime_provenance") != expected_runtime:
            raise ValueError(f"Fixture loader/runtime provenance differs from the sealed compiled candidate: {path}")
        if record.get("prompt_ids") != generation_stream[:prefix] or record.get("next_token") != generation_stream[prefix]:
            raise ValueError(f"Fixture token stream differs from the pinned generation sequence: {path}")
        records[prefix] = {"record": record, "arrays": arrays, "source": str(path)}
    if set(records) != set(PREFIXES):
        raise ValueError(f"Pinned fixture prefixes must be {PREFIXES}, got {sorted(records)}")
    return protocol, records


def compute_plan_summary(compiled_group_path: Path, ct) -> dict:
    """Record model-plan predictions only; this is not execution-device proof."""
    try:
        from coremltools.models.compute_plan import MLComputePlan
        plan = MLComputePlan.load_from_path(str(compiled_group_path), compute_units=ct.ComputeUnit.CPU_AND_GPU)
        program = plan.model_structure.program
        if program is None:
            return {"status": "unavailable", "reason": "compiled model structure has no ML Program"}
        counts = {}
        preferred = {}
        supported = {}
        cost_weights = []
        unknown = 0

        def walk(block):
            for operation in block.operations:
                yield operation
                for nested in operation.blocks:
                    yield from walk(nested)

        operations = [op for function in program.functions.values() for op in walk(function.block)]
        for operation in operations:
            kind = operation.operator_name
            counts[kind] = counts.get(kind, 0) + 1
            try:
                usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
                if usage is None:
                    unknown += 1
                    continue
                device = type(usage.preferred_compute_device).__name__
                preferred.setdefault(kind, {})[device] = preferred.setdefault(kind, {}).get(device, 0) + 1
                for supported_device in usage.supported_compute_devices:
                    name = type(supported_device).__name__
                    supported.setdefault(kind, {})[name] = supported.setdefault(kind, {}).get(name, 0) + 1
                cost = plan.get_estimated_cost_for_mlprogram_operation(operation)
                if cost is not None:
                    cost_weights.append(float(cost.weight))
            except Exception:
                unknown += 1
        return {"status": "available", "compute_units_requested": "CPU_AND_GPU",
                "operation_count": len(operations), "operation_counts": dict(sorted(counts.items())),
                "preferred_device_counts_by_operation": {k: dict(sorted(v.items())) for k, v in sorted(preferred.items())},
                "supported_device_counts_by_operation": {k: dict(sorted(v.items())) for k, v in sorted(supported.items())},
                "operations_without_plan_usage": unknown,
                "estimated_cost_weight_sum": sum(cost_weights),
                "estimated_cost_operation_count": len(cost_weights),
                "interpretation": "Framework plan prediction only; not proof of actual runtime device placement."}
    except Exception as exc:
        return {"status": "unavailable", "error": {"type": type(exc).__name__, "message": str(exc)},
                "interpretation": "No runtime device-placement conclusion can be drawn."}


def run_call(model, inputs: dict, state_before: dict, out: Path, label: str) -> dict:
    state = model.make_state()
    try:
        if set(state_before) != EXPECTED_STATES:
            raise ValueError("State fixture does not contain all expected eight entries")
        for name in sorted(EXPECTED_STATES):
            state.write_state(name, state_before[name])
        restored = {name: state.read_state(name).copy() for name in sorted(EXPECTED_STATES)}
        restoration = compare_maps(state_before, restored, EXPECTED_STATES)
        outputs = model.predict(inputs, state=state)
        states = {name: state.read_state(name).copy() for name in sorted(EXPECTED_STATES)}
        save_outputs(out, label, outputs, states)
        return {"label": label, "restoration": restoration,
                "outputs_vs_expected": None, "states_vs_expected": None,
                "outputs": outputs, "states": states}
    finally:
        del state


def compare_and_store_call(model, inputs: dict, state_before: dict, expected_outputs: dict,
                           expected_states: dict, out: Path, label: str) -> dict:
    call = run_call(model, inputs, state_before, out, label)
    out_metrics = compare_maps(expected_outputs, call["outputs"], EXPECTED_OUTPUTS)
    state_metrics = compare_maps(expected_states, call["states"], EXPECTED_STATES)
    call["outputs_vs_expected"] = out_metrics
    call["states_vs_expected"] = state_metrics
    call["strict_exact"] = out_metrics["exact"] and state_metrics["exact"] and call["restoration"]["exact"]
    del call["outputs"], call["states"]
    return call


def cold_inputs(embedding, token_ids, np):
    ids = np.asarray([token_ids], dtype=np.int32)
    embedded = embedding.predict({"ids": ids, "norm_divisor": np.asarray([1280.0], dtype=np.float32)})
    x = embedded["x"]
    ve = embedded["ve"]
    count = len(token_ids)
    inputs = {"x": x, "ve": ve,
              "positions": np.arange(count, dtype=np.int32),
              "tap0": x.copy(), "tap12": x.copy(), "tap23": x.copy(),
              "norm_divisor": np.asarray([1280.0], dtype=np.float32)}
    return inputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--model-dir", default=str(SOURCE_ROOT))
    parser.add_argument("--identity-manifest", default=str(SOURCE_MANIFEST))
    parser.add_argument("--compiled-dir", default=str(COMPILED_ROOT))
    parser.add_argument("--compiled-manifest", default=str(COMPILED_MANIFEST))
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.attempt):
        parser.error("Attempt names may contain only letters, digits, hyphens, and underscores")
    model_dir = Path(args.model_dir).expanduser().resolve(strict=True)
    identity_manifest = Path(args.identity_manifest).expanduser().resolve(strict=True)
    compiled_dir = Path(args.compiled_dir).expanduser().resolve(strict=True)
    compiled_manifest = Path(args.compiled_manifest).expanduser().resolve(strict=True)
    if model_dir != SOURCE_ROOT.resolve() or identity_manifest != SOURCE_MANIFEST.resolve() or \
            compiled_dir != COMPILED_ROOT.resolve() or compiled_manifest != COMPILED_MANIFEST.resolve():
        parser.error("This probe is pinned to the original qualified candidate and full-cache-1 compiled artifact")
    out = ROOT / "evidence/06-gpu" / args.attempt
    out.mkdir(parents=True, exist_ok=False)
    (out / "source-snapshot.py").write_bytes(Path(__file__).read_bytes())
    protocol_doc = ROOT / "evidence/06-gpu/PROTOCOL.md"
    (out / "protocol-snapshot.md").write_bytes(protocol_doc.read_bytes())
    results_path = out / "results.json"
    generation_doc = ROOT / "evidence/04-coreml/generation-comparison.json"
    EXPECTED_GENERATION_SHA = "1ee6553885123a57b8fb40b89c292eea201e148e6f3bd4d9b4cba1c057f9f3eb"
    if sha(generation_doc) != EXPECTED_GENERATION_SHA:
        parser.error("Pinned generation fixture changed")
    result = {"status": "starting", "experiment_completed": False,
              "scope": "Single compiled group-0 artifact; CPU_ONLY vs CPU_AND_GPU; no runtime integration or speedup claim.",
              "attempt": args.attempt, "compute_units": {"control": "CPU_ONLY", "diagnostic": "CPU_AND_GPU"},
              "artifact_selection": {"model_dir": str(model_dir), "identity_manifest": str(identity_manifest),
                                     "compiled_dir": str(compiled_dir), "compiled_manifest": str(compiled_manifest),
                                     "group": "0", "variant_used": False},
              "protocol_sha256": sha(protocol_doc), "five_e_protocol_sha256": FIVE_E_PROTOCOL_SHA,
              "script_sha256": sha(Path(__file__)), "host": {"platform": platform.platform(),
                                                                 "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                                 "load_average": os.getloadavg()},
              "phases": [], "calls": [], "compute_plan": None, "cpu_warm_controls": {}}

    def save():
        atomic_json(results_path, result)

    def phase(name: str, state: str):
        result["phases"].append({"phase": name, "state": state,
                                 "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        save()
        print(f"{name}: {state}", flush=True)

    save()
    models = {}
    gpu_call_errors = []
    try:
        import numpy as np
        import coremltools as ct
        from coreml_port.compiled_identity import verify_compiled_identity
        from coreml_port.package_identity import verify_candidate_identity
        from coreml_port.prediction_guard import install, stats
        from coreml_port.loader_selection import loader_provenance
        from scripts.runtime_provenance import runtime_provenance

        phase("provenance", "verifying")
        candidate_identity = verify_candidate_identity(model_dir, identity_manifest)
        compiled_identity = verify_compiled_identity(compiled_dir, compiled_manifest, model_dir, candidate_identity)
        if candidate_identity["aggregate_sha256"] != EXPECTED_CANDIDATE_ID or compiled_identity["aggregate_sha256"] != EXPECTED_COMPILED_ID:
            raise ValueError("Selected source/compiled identity differs from the pinned qualified candidate")
        milestone = json.loads(MILESTONE.read_text())
        runtime_before = runtime_provenance()
        if milestone.get("milestone_pass") is not True or runtime_before != milestone.get("current_runtime_provenance"):
            raise ValueError("Current runtime/guards differ from the passing compiled milestone")
        expected_loader = milestone["compiled_identity"]["loader_provenance"]
        if expected_loader.get("compiled_aggregate_sha256") != compiled_identity["aggregate_sha256"] or \
                expected_loader.get("source_candidate_identity") != candidate_identity["aggregate_sha256"]:
            raise ValueError("Milestone loader binding differs from the selected source/compiled identity")
        current_loader = loader_provenance(argparse.Namespace(
            compiled_dir=str(compiled_dir), compiled_manifest=str(compiled_manifest),
            tiny_prefix_replay=True, compute="CPU_ONLY", model_dir=str(model_dir),
            identity_manifest=str(identity_manifest)))
        if current_loader != expected_loader:
            raise ValueError("Current loader source/manifest provenance differs from the passing milestone")
        result["source_candidate_identity"] = candidate_identity["aggregate_sha256"]
        result["compiled_identity"] = compiled_identity["aggregate_sha256"]
        result["loader_provenance"] = expected_loader
        result["current_loader_provenance"] = current_loader
        result["runtime_provenance_before"] = runtime_before
        result["source_manifest_sha256"] = sha(identity_manifest)
        result["compiled_manifest_sha256"] = sha(compiled_manifest)
        result["pinned_milestone_sha256"] = sha(MILESTONE)
        save()

        phase("guards", "installing before model constructors")
        guard_initial = install()
        result["prediction_guard_initial"] = guard_initial
        if guard_initial.get("installed") is not True:
            raise RuntimeError("Prediction/cleanup guard installation did not report installed")
        save()

        fixture_protocol, fixtures = load_fixtures(out, expected_loader, runtime_before,
                                                   (lambda obj: obj["prompt_ids"] + obj["generated_ids"])(
                                                       json.loads(generation_doc.read_text())["reference"]))
        fixture_doc = generation_doc
        result["generation_fixture_sha256"] = sha(fixture_doc)
        result["five_e_fixture_protocol"] = {"sha256": FIVE_E_PROTOCOL_SHA,
                                              "fixtures": [{"path": p["results"], "sha256": p["sha256"]}
                                                           for p in fixture_protocol["fixtures"]]}
        save()

        group_path = compiled_dir / "0.mlmodelc"
        phase("compute_plan", "loading CPU_AND_GPU plan (prediction only; not runtime proof)")
        result["compute_plan"] = compute_plan_summary(group_path, ct)
        save()

        phase("model_loading", "loading CPU embedding and original group0 for both compute policies")
        models["embedding_cpu"] = ct.models.CompiledMLModel(
            str(compiled_dir / "embedding.mlmodelc"), compute_units=ct.ComputeUnit.CPU_ONLY)
        models["group_cpu"] = ct.models.CompiledMLModel(
            str(group_path), compute_units=ct.ComputeUnit.CPU_ONLY)
        models["group_gpu"] = ct.models.CompiledMLModel(
            str(group_path), compute_units=ct.ComputeUnit.CPU_AND_GPU)
        result["model_loading_complete"] = True
        save()

        # Construct true cold-prefill group inputs from the same pinned token
        # stream by running only the already-qualified CPU compiled embedding.
        stream = json.loads((ROOT / "evidence/04-coreml/generation-comparison.json").read_text())["reference"]
        token_stream = stream["prompt_ids"] + stream["generated_ids"]
        cold_cases = {}
        for count in (3, 32):
            name = f"cold-prefill-{count}"
            case_dir = out / name
            case_dir.mkdir(exist_ok=False)
            inputs = cold_inputs(models["embedding_cpu"], token_stream[:count], np)
            states = {state: np.zeros_like(fixtures[32]["arrays"]["state-before"][state])
                      for state in sorted(EXPECTED_STATES)}
            save_npz(case_dir / "inputs.npz", inputs)
            save_npz(case_dir / "state-before.npz", states)
            cold_cases[name] = {"count": count, "inputs": inputs, "state_before": states}
        result["cold_prefill"] = {name: {"tokens": item["count"],
                                         "input_shapes": {k: list(v.shape) for k, v in item["inputs"].items()},
                                         "input_dtypes": {k: str(v.dtype) for k, v in item["inputs"].items()},
                                         "state_dtype": str(next(iter(item["state_before"].values())).dtype)}
                                  for name, item in cold_cases.items()}
        save()

        phase("cpu_baselines", "running cold baselines and six warm sealed-fixture controls")
        cpu_expected = {}
        for name, item in cold_cases.items():
            entry = run_call(models["group_cpu"], item["inputs"], item["state_before"], out, f"{name}/baseline")
            cpu_expected[name] = {"outputs": entry["outputs"], "states": entry["states"]}
            result["calls"].append({"case": name, "arm": "CPU_ONLY", "repeat": 0,
                                    "source": "CPU compiled embedding + zero group state",
                                    "strict_exact": entry["restoration"]["exact"],
                                    "state_restoration": entry["restoration"]})
            if not entry["restoration"]["exact"]:
                raise ValueError(f"Cold CPU state restoration was not exact for {name}")
            save()
        for prefix in PREFIXES:
            f = fixtures[prefix]["arrays"]
            reference_dir = out / f"warm-{prefix}" / "sealed-reference"
            reference_dir.mkdir(parents=True, exist_ok=False)
            save_npz(reference_dir / "inputs.npz", f["inputs"])
            save_npz(reference_dir / "state-before.npz", f["state-before"])
            save_npz(reference_dir / "outputs.npz", f["outputs"])
            save_npz(reference_dir / "state-after.npz", f["state-after"])
            warm_cpu = []
            for repeat in range(2):
                label = f"warm-{prefix}/cpu-repeat-{repeat}"
                call = compare_and_store_call(models["group_cpu"], f["inputs"], f["state-before"],
                                              f["outputs"], f["state-after"], out, label)
                if not call["strict_exact"]:
                    result["calls"].append({"case": f"warm-{prefix}", "arm": "CPU_ONLY", "repeat": repeat,
                                            "strict_exact": False, "details": call})
                    save()
                    raise ValueError(f"CPU control did not exactly reproduce sealed warm fixture at {prefix}")
                warm_cpu.append(call)
                cpu_expected[f"warm-{prefix}-repeat-{repeat}"] = {"outputs": f["outputs"], "states": f["state-after"]}
                result["calls"].append({"case": f"warm-{prefix}", "arm": "CPU_ONLY", "repeat": repeat,
                                        "strict_exact": True, "outputs_vs_fixture": call["outputs_vs_expected"],
                                        "states_vs_fixture": call["states_vs_expected"],
                                        "state_restoration": call["restoration"]})
            result["cpu_warm_controls"][str(prefix)] = warm_cpu
            save()

        phase("gpu_replays", "running GPU policy twice per cold and warm case; retain mismatches")
        strict_gpu = True
        gpu_cases = {}
        for name, item in cold_cases.items():
            expected = cpu_expected[name]
            gpu_cases[name] = []
            for repeat in range(2):
                label = f"{name}/gpu-repeat-{repeat}"
                try:
                    call = compare_and_store_call(models["group_gpu"], item["inputs"], item["state_before"],
                                                  expected["outputs"], expected["states"], out, label)
                    strict_gpu &= call["strict_exact"]
                    gpu_cases[name].append(call)
                    result["calls"].append({"case": name, "arm": "CPU_AND_GPU", "repeat": repeat,
                                            "strict_exact": call["strict_exact"],
                                            "outputs_vs_cpu": call["outputs_vs_expected"],
                                            "states_vs_cpu": call["states_vs_expected"],
                                            "state_restoration": call["restoration"]})
                except Exception as exc:
                    strict_gpu = False
                    row = {"case": name, "arm": "CPU_AND_GPU", "repeat": repeat,
                           "strict_exact": False, "error": {"type": type(exc).__name__, "message": str(exc)}}
                    gpu_call_errors.append(row)
                    result["calls"].append(row)
                save()
        for prefix in PREFIXES:
            f = fixtures[prefix]["arrays"]
            name = f"warm-{prefix}"
            gpu_cases[name] = []
            for repeat in range(2):
                label = f"{name}/gpu-repeat-{repeat}"
                try:
                    call = compare_and_store_call(models["group_gpu"], f["inputs"], f["state-before"],
                                                  f["outputs"], f["state-after"], out, label)
                    strict_gpu &= call["strict_exact"]
                    gpu_cases[name].append(call)
                    result["calls"].append({"case": name, "arm": "CPU_AND_GPU", "repeat": repeat,
                                            "strict_exact": call["strict_exact"],
                                            "outputs_vs_fixture": call["outputs_vs_expected"],
                                            "states_vs_fixture": call["states_vs_expected"],
                                            "state_restoration": call["restoration"]})
                except Exception as exc:
                    strict_gpu = False
                    row = {"case": name, "arm": "CPU_AND_GPU", "repeat": repeat,
                           "strict_exact": False, "error": {"type": type(exc).__name__, "message": str(exc)}}
                    gpu_call_errors.append(row)
                    result["calls"].append(row)
                save()

        result["gpu_call_errors"] = gpu_call_errors
        result["gpu_replay_count"] = sum(len(items) for items in gpu_cases.values())
        result["cpu_state_call_count"] = 8
        result["gpu_state_call_count"] = result["gpu_replay_count"]
        result["strict_same_graph_device_parity_passed"] = bool(strict_gpu and not gpu_call_errors and
                                                                result["gpu_replay_count"] == 10)
        result["operational_status"] = "calls_completed" if not gpu_call_errors else "gpu_call_errors_recorded"
        result["execution_device_verified"] = False
        result["strict_fidelity_status"] = ("exact" if result["strict_same_graph_device_parity_passed"]
                                             else "not_exact")
        result["diagnostic_thresholds"] = {
            "relative_rmse_limit": 0.01, "cosine_minimum": 0.995,
            "status": "diagnostic only; strict exactness is the device-parity gate and these are not global acceptance thresholds"}

        models.clear()
        import gc
        gc.collect()
        phase("preservation", "rechecking source package, original compiled artifact, and runtime provenance")
        source_after = verify_candidate_identity(model_dir, identity_manifest)
        compiled_after = verify_compiled_identity(compiled_dir, compiled_manifest, model_dir, source_after)
        runtime_after = runtime_provenance()
        result["source_preserved"] = source_after["aggregate_sha256"] == EXPECTED_CANDIDATE_ID
        result["compiled_preserved"] = compiled_after["aggregate_sha256"] == EXPECTED_COMPILED_ID
        result["runtime_provenance_after"] = runtime_after
        result["runtime_preserved"] = runtime_after == runtime_before
        result["prediction_guard_final"] = stats()
        expected_guard_final = {"installed": True, "plain_calls": 2, "state_calls": 18, "gil_releases": 20}
        result["guard_counts_match_call_plan"] = result["prediction_guard_final"] == expected_guard_final
        if not (result["source_preserved"] and result["compiled_preserved"] and result["runtime_preserved"] and result["guard_counts_match_call_plan"]):
            raise RuntimeError("Source, compiled identity, runtime, or prediction-guard preservation check failed")
        stable_files = {"script": Path(__file__), "protocol": protocol_doc,
                        "five_e_protocol": FIVE_E_PROTOCOL, "generation_fixture": generation_doc}
        expected_stable = {"script": result["script_sha256"], "protocol": result["protocol_sha256"],
                           "five_e_protocol": FIVE_E_PROTOCOL_SHA, "generation_fixture": EXPECTED_GENERATION_SHA}
        result["stable_file_hashes_after"] = {name: sha(path) for name, path in stable_files.items()}
        if result["stable_file_hashes_after"] != expected_stable:
            raise RuntimeError("Script/protocol/pinned fixture changed during probe")
        inventory = []
        for path in sorted(out.rglob("*")):
            if path.is_file() and path != results_path:
                inventory.append({"path": str(path.relative_to(out)), "size_bytes": path.stat().st_size,
                                  "sha256": sha(path)})
        result["evidence_inventory"] = inventory
        gpu_parity = result["strict_same_graph_device_parity_passed"]
        result["experiment_completed"] = True
        result["status"] = "completed"
        result["decision"] = ("CPU_ONLY versus CPU_AND_GPU compute-policy exact parity observed for this bounded group0 probe; execution device unverified; no runtime integration or speedup claim"
                              if gpu_parity else "CPU_ONLY versus CPU_AND_GPU compute-policy exactness no-go; retain diagnostic captures; execution device unverified; no runtime integration or speedup claim")
        result["performance_claim"] = None
        phase("complete", "compute-policy exact parity" if gpu_parity else "completed compute-policy exactness no-go; mismatch captures retained")
    except BaseException as exc:
        result.update(status="failed", experiment_completed=False,
                      failure={"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        save()
        print(f"failed: {type(exc).__name__}: {exc}", flush=True)
        return 1
    finally:
        models.clear()
        result["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save()
    print(json.dumps({"status": result["status"], "decision": result.get("decision"),
                      "strict_same_graph_device_parity_passed": result.get("strict_same_graph_device_parity_passed"),
                      "gpu_replay_count": result.get("gpu_replay_count"),
                      "compute_plan_status": (result.get("compute_plan") or {}).get("status")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
