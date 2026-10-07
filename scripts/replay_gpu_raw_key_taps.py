"""Replay saved group-0 CPU inputs/cache through original and raw-tapped models.

Eight stateful calls: two chunks, two compute policies, two independently
restored model variants. Within-policy endpoint and all-state equality is a
hard prerequisite for interpreting any tapped intermediate.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE = ROOT / "artifacts/04d-cpu/full-candidate-6"
SOURCE_MANIFEST = ROOT / "evidence/04d-cpu/full-candidate-6/package-identity.json"
CACHE = ROOT / "artifacts/05c-compiled/full-cache-1"
CACHE_MANIFEST = ROOT / "evidence/05c-compiled/full-cache-1/compiled-identity.json"
SOURCE_ID = "5a3e4c7a571e4fc3532377d9b9bb0da023f499c10709216fee7e1001bf4ad2c6"
CACHE_ID = "a838dd36e8f87c6b9fc0dd7865d0912651c6ab3399fa299d10ff12d788bcb031"
TAPS = ("x_5", "x_9", "value_5", "k_7", "x_27", "var_2083",
        "x_29", "var_2106", "x_31", "step_23", "var_2121", "var_2122", "k_9")
PREVIOUS_TAPS = ("x_5", "value_5", "k_7", "k_9")
ENDPOINTS = ("y", "tap23_out")
INPUTS = {"x", "ve", "positions", "tap0", "tap12", "tap23", "norm_divisor"}
STATES = tuple(f"inner_{kind}_cache_{layer}"
               for layer in range(4) for kind in ("k", "v"))


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected object: {path}")
    return value


def save(path: Path, value: dict) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")
    temp.replace(path)


def inventory(root: Path) -> dict[str, dict[str, object]]:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlink in sealed tree: {path}")
        if path.is_file():
            result[str(path.relative_to(root))] = {"size": path.stat().st_size,
                                                   "sha256": sha(path)}
        elif not path.is_dir():
            raise ValueError(f"Nonregular entry in sealed tree: {path}")
    return result


def equal(a, b) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and \
        a.tobytes(order="C") == b.tobytes(order="C")


def first_difference(a, b) -> list[int] | None:
    import numpy as np
    if a.shape != b.shape or a.dtype != b.dtype:
        return None
    a_bytes = np.frombuffer(a.tobytes(order="C"), dtype=np.uint8)
    b_bytes = np.frombuffer(b.tobytes(order="C"), dtype=np.uint8)
    different = np.argwhere(a_bytes != b_bytes)
    if not len(different):
        return None
    # Report the first differing value's coordinate, even for byte-level
    # signed-zero or NaN payload differences.
    scalar_index = int(different[0, 0]) // a.dtype.itemsize
    return [int(i) for i in np.unravel_index(scalar_index, a.shape)]


def compare_maps(actual: dict, expected: dict) -> dict:
    if set(actual) != set(expected):
        raise ValueError(f"Array names differ: {set(actual) ^ set(expected)}")
    return {name: {"exact": equal(actual[name], expected[name]),
                   "shape": list(actual[name].shape), "dtype": str(actual[name].dtype),
                   "first_difference": first_difference(actual[name], expected[name])}
            for name in sorted(actual)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args()
    if not args.attempt or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in args.attempt):
        parser.error("Fresh lowercase path-safe --attempt required")
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = read(protocol_path)
    if protocol.get("schema_version") != 1 or protocol.get("script_sha256") != sha(Path(__file__)):
        raise ValueError("Replay protocol or script hash differs")
    if protocol.get("source_result") != "evidence/06c-gpu-localization/prompt0-1/results.json" or \
       protocol.get("builder_result") != "evidence/06c-gpu-localization/raw-key-taps-1/results.json" or \
       protocol.get("chunks") != [0, 1] or protocol.get("compute_policies") != ["CPU", "GPU"] or \
       protocol.get("taps") != list(TAPS) or protocol.get("call_count") != 8:
        raise ValueError("Replay protocol scope differs")
    if sha(ROOT / protocol["source_result"]) != protocol["source_result_sha256"]:
        raise ValueError("Pinned prompt-localization result changed")
    if sha(ROOT / protocol["builder_result"]) != protocol["builder_result_sha256"]:
        raise ValueError("Pinned tap-builder result changed")
    if protocol.get("previous_tap_result") != "evidence/06c-gpu-localization/key-taps-replay-1/results.json" or \
       sha(ROOT / protocol["previous_tap_result"]) != protocol.get("previous_tap_result_sha256"):
        raise ValueError("Previous four-tap replay evidence changed")
    if protocol.get("builder_script_sha256") != sha(ROOT / "scripts/build_gpu_raw_key_taps.py"):
        raise ValueError("Tapped builder source changed")
    output = ROOT / "evidence/06c-gpu-localization" / args.attempt
    if output.exists():
        raise FileExistsError("Fresh replay evidence attempt required")
    output.mkdir(parents=True, exist_ok=False)
    (output / "source-snapshot.py").write_bytes(Path(__file__).read_bytes())
    (output / "protocol.json").write_bytes(protocol_path.read_bytes())
    report = {"status": "starting", "passed": False, "script_sha256": sha(Path(__file__)),
              "protocol_sha256": sha(protocol_path), "source_result_sha256": protocol["source_result_sha256"],
              "builder_result_sha256": None, "calls": [], "files": {},
              "actual_device_placement": "unverified; compute policy only",
              "interpretation_gate": "Original-vs-tapped endpoints and all eight states exact within each policy and chunk"}

    def update() -> None:
        save(output / "results.json", report)

    update()
    model = None
    try:
        import coremltools as ct
        import numpy as np
        from coreml_port.package_identity import verify_candidate_identity
        from coreml_port.compiled_identity import verify_compiled_identity
        from coreml_port.prediction_guard import install, stats
        from scripts.runtime_provenance import runtime_provenance

        builder_path = ROOT / protocol["builder_result"]
        builder_hash = sha(builder_path)
        assert builder_hash == protocol["builder_result_sha256"]
        built = read(builder_path)
        assert built.get("status") == "completed" and built.get("passed") is True
        assert built.get("source_candidate_identity") == SOURCE_ID
        assert built.get("script_sha256") == protocol["builder_script_sha256"]
        assert sha(ROOT / "evidence/06c-gpu-localization/raw-key-taps-1/source-snapshot.py") == built["script_sha256"]
        graph = built["graph_check"]
        assert graph["passed"] and graph["all_original_operations_byte_identical"] and \
            graph["only_description_and_block_outputs_changed"]
        assert graph["original_operations_sha256"] == graph["variant_operations_sha256"]
        assert tuple(tap["name"] for tap in graph["added_taps"]) == TAPS
        assert graph["original_outputs_preserved"] == list(ENDPOINTS) and \
            tuple(graph["state_names_preserved"]) == STATES
        assert built["package"] == "artifacts/06c-gpu-localization/raw-key-taps-1/0.mlpackage" and \
            built["compiled"] == "artifacts/06c-gpu-localization/raw-key-taps-1/0.mlmodelc"
        source_group = SOURCE / "0.mlpackage"
        tap_group = ROOT / built["package"]
        tap_compiled = ROOT / built["compiled"]
        assert inventory(source_group) == built["source_files"]
        assert inventory(tap_group) == built["variant_files"]
        assert inventory(tap_compiled) == built["compiled_files"]
        source_identity = verify_candidate_identity(SOURCE, SOURCE_MANIFEST)
        compiled_identity = verify_compiled_identity(CACHE, CACHE_MANIFEST, SOURCE, SOURCE_MANIFEST)
        assert source_identity["aggregate_sha256"] == SOURCE_ID and \
            compiled_identity["aggregate_sha256"] == CACHE_ID
        assert tuple(compiled_identity["packages"]["0"]["state_names"]) == STATES
        assert compiled_identity["packages"]["0"]["output_names"] == list(ENDPOINTS)
        report.update(builder_result_sha256=builder_hash,
                      source_candidate_identity=SOURCE_ID, original_compiled_identity=CACHE_ID,
                      tapped_package=built["package"], tapped_compiled=built["compiled"],
                      graph_check=graph, source_files=built["source_files"],
                      variant_files=built["variant_files"], compiled_files=built["compiled_files"])
        update()
        localization = read(ROOT / protocol["source_result"])
        assert localization["status"] == "completed" and localization.get("passed") is True
        assert localization["source_identity"] == SOURCE_ID and localization["compiled_identity"] == CACHE_ID
        assert localization["script_sha256"] == protocol["localizer_script_sha256"]
        assert sha(ROOT / "scripts/localize_gpu_drift.py") == protocol["localizer_script_sha256"]
        report["runtime_provenance"] = runtime_provenance()
        assert report["runtime_provenance"] == localization["runtime_provenance"]
        previous = read(ROOT / protocol["previous_tap_result"])
        assert previous["status"] == "completed" and previous["passed"] and \
            previous["instrumentation_valid"] and previous["prediction_guard"] == \
            {"installed": True, "plain_calls": 0, "state_calls": 8, "gil_releases": 8}

        def read_saved(label: str, expected_names: set[str]) -> dict:
            info = localization["files"][label]
            path = ROOT / info["path"]
            if sha(path) != info["sha256"]:
                raise ValueError(f"Saved source archive changed: {label}")
            with np.load(path) as stored:
                arrays = {name: stored[name].copy() for name in stored.files}
            if set(arrays) != expected_names:
                raise ValueError(f"Saved source array set differs: {label}")
            assert {name: list(value.shape) for name, value in arrays.items()} == info["shapes"]
            assert {name: str(value.dtype) for name, value in arrays.items()} == info["dtypes"]
            return arrays

        def archive(label: str, arrays: dict) -> dict:
            path = output / (label + ".npz")
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **arrays)
            info = {"path": str(path.relative_to(ROOT)), "sha256": sha(path),
                    "shapes": {name: list(value.shape) for name, value in arrays.items()},
                    "dtypes": {name: str(value.dtype) for name, value in arrays.items()}}
            report["files"][label] = info
            return info

        saved = {}
        for chunk in (0, 1):
            prefix = f"chunk{chunk}/group0/own-CPU"
            inputs = read_saved(prefix + "/inputs", INPUTS)
            before = read_saved(prefix + "/state-before", set(STATES))
            assert len(inputs["positions"]) == (32 if chunk == 0 else 12)
            assert np.array_equal(inputs["positions"], np.arange(0 if chunk == 0 else 32,
                                                                  32 if chunk == 0 else 44,
                                                                  dtype=np.int32))
            saved[chunk] = {"inputs": inputs, "before": before,
                            "expected": {policy: {
                                "outputs": read_saved(f"chunk{chunk}/group0/replay-ICPU-SCPU-P{policy}/outputs",
                                                      set(ENDPOINTS)),
                                "after": read_saved(f"chunk{chunk}/group0/replay-ICPU-SCPU-P{policy}/state-after",
                                                    set(STATES))}
                                for policy in ("CPU", "GPU")}}
            archive(f"chunk{chunk}/effective-inputs", inputs)
            archive(f"chunk{chunk}/effective-state-before", before)
        install()
        assert stats() == {"installed": True, "plain_calls": 0,
                           "state_calls": 0, "gil_releases": 0}
        calls = {}
        restored = 0
        disposed = 0
        for policy in ("CPU", "GPU"):
            units = ct.ComputeUnit.CPU_ONLY if policy == "CPU" else ct.ComputeUnit.CPU_AND_GPU
            for kind, compiled in (("original", CACHE / "0.mlmodelc"),
                                   ("tapped", tap_compiled)):
                started = time.perf_counter()
                model = ct.models.CompiledMLModel(str(compiled), compute_units=units)
                report.setdefault("load_seconds", {})[f"{policy}-{kind}"] = time.perf_counter() - started
                update()
                for chunk in (0, 1):
                    state = model.make_state()
                    try:
                        for name, value in saved[chunk]["before"].items():
                            state.write_state(name, value.copy())
                        got_before = {name: state.read_state(name).copy() for name in STATES}
                        assert all(equal(got_before[name], saved[chunk]["before"][name]) for name in STATES)
                        restored += 1
                        payload = {name: value.copy() for name, value in saved[chunk]["inputs"].items()}
                        result = model.predict(payload, state=state)
                        outputs = {name: value.copy() for name, value in result.items()}
                        expected_outputs = set(ENDPOINTS) | (set(TAPS) if kind == "tapped" else set())
                        assert set(outputs) == expected_outputs
                        after = {name: state.read_state(name).copy() for name in STATES}
                        assert all(equal(payload[name], saved[chunk]["inputs"][name]) for name in INPUTS)
                        label = f"chunk{chunk}/{policy}-{kind}"
                        output_info = archive(label + "/outputs", outputs)
                        state_info = archive(label + "/state-after", after)
                        calls[(chunk, policy, kind)] = {"outputs": outputs, "after": after}
                        row = {"chunk": chunk, "policy": policy, "model": kind,
                               "state_restored_exact": True, "input_unchanged": True,
                               "outputs": output_info, "state_after": state_info}
                        if kind == "original":
                            row["saved_factorial_outputs"] = compare_maps(outputs, saved[chunk]["expected"][policy]["outputs"])
                            row["saved_factorial_state_after"] = compare_maps(after, saved[chunk]["expected"][policy]["after"])
                        report["calls"].append(row)
                        update()
                    finally:
                        del state
                        disposed += 1
                del model
                model = None
                gc.collect()
        assert len(calls) == restored == disposed == 8
        report["fresh_states"] = {"restored_exact": restored, "python_references_deleted": disposed}
        report["prediction_guard"] = stats()
        assert stats() == {"installed": True, "plain_calls": 0,
                           "state_calls": 8, "gil_releases": 8}
        controls = []
        for chunk in (0, 1):
            for policy in ("CPU", "GPU"):
                original = calls[(chunk, policy, "original")]
                tapped = calls[(chunk, policy, "tapped")]
                endpoint = compare_maps({name: tapped["outputs"][name] for name in ENDPOINTS},
                                        original["outputs"])
                states = compare_maps(tapped["after"], original["after"])
                saved_row = next(row for row in report["calls"] if row["chunk"] == chunk and
                                 row["policy"] == policy and row["model"] == "original")
                source_ok = all(item["exact"] for item in (*saved_row["saved_factorial_outputs"].values(),
                                                           *saved_row["saved_factorial_state_after"].values()))
                controls.append({"chunk": chunk, "policy": policy, "endpoint": endpoint,
                                 "states": states, "original_matches_saved_factorial": source_ok,
                                 "passed": source_ok and all(item["exact"] for item in (*endpoint.values(), *states.values()))})
        report["same_policy_controls"] = controls
        report["instrumentation_valid"] = all(row["passed"] for row in controls)
        if report["instrumentation_valid"]:
            shared_checks = []
            for chunk in (0, 1):
                for policy in ("CPU", "GPU"):
                    label = f"chunk{chunk}/{policy}-tapped/outputs"
                    info = previous["files"][label]
                    prior_path = ROOT / info["path"]
                    assert sha(prior_path) == info["sha256"]
                    with np.load(prior_path) as archived:
                        assert set(archived.files) == set(ENDPOINTS) | set(PREVIOUS_TAPS)
                        prior = {name: archived[name].copy() for name in PREVIOUS_TAPS}
                    current = {name: calls[(chunk, policy, "tapped")]["outputs"][name]
                               for name in PREVIOUS_TAPS}
                    shared_checks.append({"chunk": chunk, "policy": policy,
                                          "comparisons": compare_maps(current, prior)})
            report["previous_tap_checks"] = shared_checks
            report["previous_tap_compatible"] = all(
                item["exact"] for check in shared_checks for item in check["comparisons"].values())
            report["tap_policy_comparisons"] = [
                {"chunk": chunk,
                 "comparisons": compare_maps(
                     {name: calls[(chunk, "GPU", "tapped")]["outputs"][name] for name in TAPS},
                     {name: calls[(chunk, "CPU", "tapped")]["outputs"][name] for name in TAPS})}
                for chunk in (0, 1)]
        else:
            report["previous_tap_checks"] = None
            report["previous_tap_compatible"] = None
            report["tap_policy_comparisons"] = None
        assert verify_candidate_identity(SOURCE, SOURCE_MANIFEST) == source_identity
        assert verify_compiled_identity(CACHE, CACHE_MANIFEST, SOURCE, SOURCE_MANIFEST) == compiled_identity
        assert inventory(source_group) == built["source_files"] and \
            inventory(tap_group) == built["variant_files"] and \
            inventory(tap_compiled) == built["compiled_files"]
        assert sha(builder_path) == builder_hash and sha(Path(__file__)) == report["script_sha256"] and \
            sha(protocol_path) == report["protocol_sha256"] and \
            sha(ROOT / protocol["source_result"]) == report["source_result_sha256"] and \
            sha(ROOT / protocol["previous_tap_result"]) == protocol["previous_tap_result_sha256"] and \
            sha(ROOT / "scripts/localize_gpu_drift.py") == protocol["localizer_script_sha256"] and \
            runtime_provenance() == report["runtime_provenance"] and \
            sha(ROOT / "scripts/build_gpu_raw_key_taps.py") == protocol["builder_script_sha256"] and \
            sha(ROOT / "evidence/06c-gpu-localization/raw-key-taps-1/source-snapshot.py") == \
            protocol["builder_script_sha256"]
        assert report["instrumentation_valid"], "Tapped package changed an original endpoint/state or baseline replay"
        report.update(status="completed", passed=True, diagnostic_completed=True,
                      gpu_numerical_qualification="not_qualified")
    except BaseException as exc:
        report.update(status="failed", passed=False,
                      error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        model = None
        gc.collect()
        update()
    print(json.dumps({"passed": report["passed"],
                      "instrumentation_valid": report["instrumentation_valid"],
                      "prediction_guard": report["prediction_guard"]}, indent=2))


if __name__ == "__main__":
    main()
