"""Graft the native-tested external-seed reciprocal MIL block into group 0.

The correction operations come verbatim from native-1's saved MIL package,
apart from necessary SSA names and symbolic tensor dimensions. A separate
bounded-domain fallback keeps the old reciprocal seed outside [0.5, 8].
Compilation is optional and reserved for the root's launch.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE_RESULT = ROOT / "evidence/06d-gpu-rounding/group0-1/results.json"
TAPPED_RESULT = ROOT / "evidence/06e-gpu-normalization/taps-1/results.json"
HOST_RESULT = ROOT / "evidence/06f-gpu-reciprocal/host-1/results.json"
NATIVE_RESULT = ROOT / "evidence/06f-gpu-reciprocal/native-1/results.json"
HELPER = ROOT / "scripts/boundary_reciprocal.py"
SOURCE_RESULT_SHA = "4e6ac1baa2f70c10bf812523f7e341d9dabf34b30299e4819631f2db31cf150f"
TAPPED_RESULT_SHA = "b2b08da42ff1e35c693211e3869af3ea7475e7e3c5e3218a3a18c3a2568e629d"
HOST_RESULT_SHA = "b3bdac0b56edaf158c0d83fbc379b0144eb5f5d4379ec58759d7c5029752f076"
NATIVE_RESULT_SHA = "ebc588038c8b3bf6273e6b1596ac5fbb619174826a0651e10d2141d49d93b026"
HELPER_SHA = "4edabf62871e35c834185e586e2f7253f6feeb48a1db9f8daae4596d9471e263"
SOURCE_OPS = 46035
TARGET_INDEX = 9543
TARGET_OUTPUT = "var_5675"
SEED_OUTPUT = "reciprocal_seed"
CORRECTED_OUTPUT = "boundary_corrected"
MODEL_FILE = "Data/com.apple.CoreML/model.mlmodel"
GUARD_TYPES = ("const", "const", "less", "greater", "logical_or", "select")


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def wire(message) -> bytes:
    return message.SerializeToString(deterministic=True)


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def inventory(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlink in package: {path}")
        if path.is_file():
            result[str(path.relative_to(root))] = sha(path)
        elif not path.is_dir():
            raise ValueError(f"Nonregular package entry: {path}")
    return result


def read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def block(spec):
    assert set(spec.mlProgram.functions) == {"main"}
    function = spec.mlProgram.functions["main"]
    assert set(function.block_specializations) == {"CoreML8"}
    return function.block_specializations["CoreML8"]


def names_in(op) -> list[str]:
    return [argument.name for binding in op.inputs.values()
            for argument in binding.arguments if argument.name]


def set_input(op, binding_name: str, name: str) -> None:
    arguments = op.inputs[binding_name].arguments
    assert len(arguments) == 1 and arguments[0].name
    arguments[0].name = name


def set_output(op, name: str, dimensions=None) -> None:
    assert len(op.outputs) == 1 and "name" in op.attributes
    op.outputs[0].name = name
    labels = op.attributes["name"].immediateValue.tensor.strings.values
    assert len(labels) == 1
    labels[0] = name
    if dimensions is not None:
        tensor = op.outputs[0].type.tensorType
        tensor.rank = len(dimensions)
        del tensor.dimensions[:]
        tensor.dimensions.extend(dimensions)


def context():
    for path, expected in ((SOURCE_RESULT, SOURCE_RESULT_SHA), (TAPPED_RESULT, TAPPED_RESULT_SHA),
                           (HOST_RESULT, HOST_RESULT_SHA), (NATIVE_RESULT, NATIVE_RESULT_SHA),
                           (HELPER, HELPER_SHA)):
        if sha(path) != expected:
            raise ValueError(f"Frozen input changed: {path}")
    source, tapped, host, native = map(read, (SOURCE_RESULT, TAPPED_RESULT, HOST_RESULT, NATIVE_RESULT))
    assert source["status"] == tapped["status"] == host["status"] == native["status"] == "completed"
    assert source["passed"] and tapped["passed"] and host["passed"]
    assert host["corrections"] == 31147 and not host["runtime_fp64"]
    assert host["pins"][str(HELPER.relative_to(ROOT))] == HELPER_SHA
    assert native["seeded_component_passed"]
    assert native["gates"]["C3b_external_seed_arithmetic"]
    assert native["gates"]["cpu_control_oracle"] and native["gates"]["cpu_preservation"]
    assert native["pins"][str(HELPER.relative_to(ROOT))] == HELPER_SHA
    assert not native["baseline_reproduced"] and not native["gates"]["C3_saved_baseline_reproduced"]
    assert source["source_candidate_identity"] == tapped["source_candidate_identity"]
    assert tapped["source_build_sha256"] == SOURCE_RESULT_SHA
    assert tapped["source_package_files"] == source["variant_files"]
    source_package = ROOT / source["package"]
    tapped_package = ROOT / tapped["package"]
    correction_record = native["models"]["correction"]
    correction_package = ROOT / correction_record["package"]
    assert inventory(source_package) == source["variant_files"]
    assert inventory(tapped_package) == tapped["variant_files"]
    assert inventory(correction_package) == correction_record["package_files"]
    return source, tapped, host, native, source_package, tapped_package, correction_package


def inspect_specs(source_spec, tapped_spec, correction_spec, native) -> dict:
    original = list(block(source_spec).operations)
    tapped_ops = list(block(tapped_spec).operations)
    correction = list(block(correction_spec).operations)
    assert len(original) == len(tapped_ops) == SOURCE_OPS
    assert all(wire(a) == wire(b) for a, b in zip(original, tapped_ops))
    assert all(not op.blocks for op in original + correction)
    target = original[TARGET_INDEX]
    assert target.type == "real_div" and len(target.outputs) == 1
    assert target.outputs[0].name == TARGET_OUTPUT and target.outputs[0].type.tensorType.dataType == 11
    assert {key: [a.name for a in binding.arguments] for key, binding in target.inputs.items()} == \
        {"x": ["fill_like_4"], "y": ["denom_7"]}
    assert sum(v.name == TARGET_OUTPUT for op in original for v in op.outputs) == 1
    consumers = [(i, op.type, [v.name for v in op.outputs]) for i, op in enumerate(original)
                 if TARGET_OUTPUT in names_in(op)]
    assert consumers == [(TARGET_INDEX + 1, "mul", ["x_67"])]
    assert not any(v.name in (SEED_OUTPUT, CORRECTED_OUTPUT) for op in original for v in op.outputs)
    input_names = [v.name for v in correction_spec.mlProgram.functions["main"].inputs]
    assert input_names == ["d", "seed"] and list(block(correction_spec).outputs) == ["value"]
    assert len(correction) == sum(native["models"]["correction"]["operation_counts"].values())
    assert digest(b"".join(wire(op) for op in correction)) == \
        native["models"]["correction"]["operations_sha256"]
    assert not any(op.type in ("real_div", "inverse") for op in correction)
    assert all(len(op.outputs) == 1 for op in correction)
    taps = [entry.name for entry in native_taps(tapped_spec, source_spec)]
    assert len(taps) == 12
    return {"source_operation_count": len(original), "target_index": TARGET_INDEX,
            "target_operation_sha256": digest(wire(target)), "target_output": TARGET_OUTPUT,
            "seed_output": SEED_OUTPUT, "target_consumers": consumers,
            "source_operation_sha256": digest(b"".join(wire(op) for op in original)),
            "correction_operation_count": len(correction),
            "correction_operation_sha256": digest(b"".join(wire(op) for op in correction)),
            "correction_input_names": input_names, "taps": taps,
            "other_source_operations_byte_identical_to_tapped_source": True,
            "native_component_gate_passed": native["seeded_component_passed"],
            "standalone_seed_reproduction_passed": native["baseline_reproduced"]}


def native_taps(tapped_spec, source_spec):
    old = [item.name for item in source_spec.description.output]
    assert old == list(block(source_spec).outputs) == ["y", "tap23_out"]
    entries = list(tapped_spec.description.output)
    assert [entry.name for entry in entries[:2]] == old
    names = [entry.name for entry in entries[2:]]
    assert names == list(block(tapped_spec).outputs)[2:]
    assert len(names) == 12 and len(set(names)) == 12
    return entries[2:]


def source_preflight() -> dict:
    import coremltools as ct

    source, tapped, host, native, source_package, tapped_package, correction_package = context()
    source_spec = ct.models.utils.load_spec(str(source_package))
    tapped_spec = ct.models.utils.load_spec(str(tapped_package))
    correction_spec = ct.models.utils.load_spec(str(correction_package))
    check = inspect_specs(source_spec, tapped_spec, correction_spec, native)
    plain_spec, rewrite_check = rewrite(source_spec, correction_spec)
    candidate_tapped_spec, tapped_check = add_taps(plain_spec, tapped_spec, source_spec)
    assert wire(block(plain_spec).operations[0]) == wire(block(candidate_tapped_spec).operations[0])
    return {"status": "completed", "passed": True,
            "builder_source_sha256": sha(Path(__file__)),
            "source_result_sha256": SOURCE_RESULT_SHA,
            "tapped_result_sha256": TAPPED_RESULT_SHA,
            "host_result_sha256": HOST_RESULT_SHA,
            "native_result_sha256": NATIVE_RESULT_SHA,
            "helper_sha256": HELPER_SHA,
            "source_candidate_identity": source["source_candidate_identity"],
            "source_graph_check": check,
            "graph_check": rewrite_check,
            "tapped_graph_check": tapped_check,
            "build_status": "not built; static saved-spec inspection only",
            "native_scope": "C3b external-seed component passed; C3 standalone GPU seed reproduction failed and requires in-model replay"}


def remap_exported(correction_ops, original_type):
    names = {"d": "denom_7", "seed": SEED_OUTPUT}
    for op in correction_ops:
        old_name = op.outputs[0].name
        names[old_name] = CORRECTED_OUTPUT if old_name == "value" else f"boundary_site_{TARGET_INDEX}_{old_name}"
    assert len(names) == len(correction_ops) + 2
    rewritten = []
    for old in correction_ops:
        new = copy.deepcopy(old)
        for binding in new.inputs.values():
            for argument in binding.arguments:
                if argument.name:
                    argument.name = names[argument.name]
        output = old.outputs[0]
        dimensions = original_type.tensorType.dimensions if old.type != "const" else None
        set_output(new, names[output.name], dimensions)
        rewritten.append(new)
    assert rewritten[-1].outputs[0].name == CORRECTED_OUTPUT
    return rewritten, names


def guard_from_export(correction_ops, original_output):
    scalar = next(op for op in correction_ops if op.type == "const" and
                  op.outputs[0].type.tensorType.rank == 0 and
                  op.outputs[0].type.tensorType.dataType == 11 and
                  len(op.attributes["val"].immediateValue.tensor.floats.values) == 1)
    by_type = {name: next(op for op in correction_ops if op.type == name)
               for name in ("less", "greater", "logical_or", "select")}
    dimensions = original_output.type.tensorType.dimensions
    produced = []
    for name, value in (("boundary_guard_min", 0.5), ("boundary_guard_max", 8.0)):
        op = copy.deepcopy(scalar)
        set_output(op, name)
        op.attributes["val"].immediateValue.tensor.floats.values[:] = [value]
        produced.append(op)
    for kind, name, left, right in (("less", "boundary_guard_below", "denom_7", "boundary_guard_min"),
                                    ("greater", "boundary_guard_above", "denom_7", "boundary_guard_max"),
                                    ("logical_or", "boundary_guard_outside", "boundary_guard_below", "boundary_guard_above")):
        op = copy.deepcopy(by_type[kind])
        set_input(op, "x", left)
        set_input(op, "y", right)
        set_output(op, name, dimensions)
        produced.append(op)
    final = copy.deepcopy(by_type["select"])
    set_input(final, "cond", "boundary_guard_outside")
    set_input(final, "a", SEED_OUTPUT)
    set_input(final, "b", CORRECTED_OUTPUT)
    set_output(final, TARGET_OUTPUT, dimensions)
    final.outputs[0].CopyFrom(original_output)
    produced.append(final)
    assert tuple(op.type for op in produced) == GUARD_TYPES
    return produced


def rewrite(source_spec, correction_spec) -> tuple[object, dict]:
    original = list(block(source_spec).operations)
    target = original[TARGET_INDEX]
    assert target.type == "real_div" and target.outputs[0].name == TARGET_OUTPUT
    correction_ops = list(block(correction_spec).operations)
    variant = copy.deepcopy(source_spec)
    new_block = block(variant)
    seed = copy.deepcopy(target)
    set_output(seed, SEED_OUTPUT)
    exported, mapping = remap_exported(correction_ops, target.outputs[0].type)
    guard = guard_from_export(correction_ops, target.outputs[0])
    rebuilt = [copy.deepcopy(op) for op in original[:TARGET_INDEX]] + \
        [seed] + exported + guard + [copy.deepcopy(op) for op in original[TARGET_INDEX + 1:]]
    del new_block.operations[:]
    new_block.operations.extend(rebuilt)
    assert len(rebuilt) == SOURCE_OPS + len(exported) + len(guard)
    assert sum(op.type == "real_div" for op in rebuilt) == \
        sum(op.type == "real_div" for op in original)
    unchanged = [(i, i if i < TARGET_INDEX else i + len(exported) + len(guard))
                 for i in range(SOURCE_OPS) if i != TARGET_INDEX]
    assert len(unchanged) == SOURCE_OPS - 1
    assert all(wire(original[i]) == wire(rebuilt[j]) for i, j in unchanged)
    assert set(names_in(rebuilt[TARGET_INDEX])) == {"fill_like_4", "denom_7"}
    assert rebuilt[TARGET_INDEX].outputs[0].name == SEED_OUTPUT
    assert rebuilt[TARGET_INDEX + 1 + len(exported) + len(guard)].type == "mul"
    assert names_in(rebuilt[TARGET_INDEX + 1 + len(exported) + len(guard)]) == \
        names_in(original[TARGET_INDEX + 1])
    existing = {entry.name for entry in source_spec.mlProgram.functions["main"].inputs}
    for op in rebuilt:
        assert set(names_in(op)) <= existing, (op.type, sorted(set(names_in(op)) - existing))
        for output in op.outputs:
            assert output.name not in existing
            existing.add(output.name)
    assert set(new_block.outputs) <= existing
    shell_before, shell_after = copy.deepcopy(source_spec), copy.deepcopy(variant)
    del block(shell_before).operations[:]
    del block(shell_after).operations[:]
    assert wire(shell_before) == wire(shell_after)
    check = {"passed": True, "original_operation_count": SOURCE_OPS,
             "untouched_operation_count": len(unchanged),
             "all_other_operations_byte_identical": True,
             "original_real_div_output_renamed_to": SEED_OUTPUT,
             "original_real_div_source_sha256": digest(wire(target)),
             "exported_correction_operation_count": len(exported),
             "exported_correction_operations_sha256": digest(b"".join(wire(op) for op in correction_ops)),
             "exported_mapping": mapping,
             "guard_operation_types": list(GUARD_TYPES),
             "guard_operation_count": len(guard),
             "bounded_denominator_range_inclusive": [0.5, 8.0],
             "outside_range_returns_original_seed": True,
             "original_output_descriptor_preserved": wire(guard[-1].outputs[0]) == wire(target.outputs[0]),
             "all_nonoperation_metadata_identical": True,
             "state_operations_and_weights_unchanged": True,
             "topological_references_valid": True,
             "seed_accuracy_scope": "One-ULP seed assumption verified on tested fixtures only; in-model GPU seed reproduction pending"}
    assert check["original_output_descriptor_preserved"]
    return variant, check


def add_taps(plain_spec, tapped_source_spec, original_source_spec):
    tapped = copy.deepcopy(plain_spec)
    names = [entry.name for entry in native_taps(tapped_source_spec, original_source_spec)]
    for entry in list(tapped_source_spec.description.output)[2:]:
        tapped.description.output.add().CopyFrom(entry)
        block(tapped).outputs.append(entry.name)
    template = next(entry for entry in tapped.description.output if entry.name == TARGET_OUTPUT)
    seed_descriptor = tapped.description.output.add()
    seed_descriptor.CopyFrom(template)
    seed_descriptor.name = SEED_OUTPUT
    block(tapped).outputs.append(SEED_OUTPUT)
    assert [entry.name for entry in tapped.description.output] == \
        ["y", "tap23_out"] + names + [SEED_OUTPUT]
    assert list(block(tapped).outputs) == [entry.name for entry in tapped.description.output]
    assert all(wire(a) == wire(b) for a, b in zip(block(plain_spec).operations, block(tapped).operations))
    assert wire(plain_spec.mlProgram.functions["main"].inputs[0]) == \
        wire(tapped.mlProgram.functions["main"].inputs[0])
    check = {"passed": True, "plain_and_tapped_operations_byte_identical": True,
             "plain_and_tapped_weights_identical": True,
             "original_endpoints_preserved": True,
             "normalization_taps": names, "seed_name": SEED_OUTPUT,
             "total_taps": len(names) + 1,
             "tapped_output_names": list(block(tapped).outputs)}
    return tapped, check


def build(attempt: str, compile_models: bool) -> dict:
    import coremltools as ct

    if not attempt or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in attempt):
        raise ValueError("Fresh path-safe attempt required")
    source, tapped_record, host, native, source_package, tapped_package, correction_package = context()
    evidence = ROOT / "evidence/06f-gpu-reciprocal" / attempt
    artifacts = ROOT / "artifacts/06f-gpu-reciprocal" / attempt
    if evidence.exists() or artifacts.exists():
        raise FileExistsError("Fresh evidence/artifact attempt required")
    evidence.mkdir(parents=True)
    artifacts.mkdir(parents=True)
    for name, path in (("source-snapshot.py", Path(__file__)), ("helper-snapshot.py", HELPER),
                       ("source-build-snapshot.json", SOURCE_RESULT),
                       ("tapped-source-snapshot.json", TAPPED_RESULT),
                       ("host-snapshot.json", HOST_RESULT), ("native-snapshot.json", NATIVE_RESULT)):
        (evidence / name).write_bytes(path.read_bytes())
    report = {"status": "starting", "passed": False, "attempt": attempt,
              "script_sha256": sha(Path(__file__)), "source_result_sha256": SOURCE_RESULT_SHA,
              "tapped_result_sha256": TAPPED_RESULT_SHA, "host_result_sha256": HOST_RESULT_SHA,
              "native_result_sha256": NATIVE_RESULT_SHA, "helper_sha256": HELPER_SHA,
              "source_candidate_identity": source["source_candidate_identity"],
              "standalone_seed_reproduction_passed": native["baseline_reproduced"],
              "seeded_component_passed": native["seeded_component_passed"],
              "numerical_qualification": "Only external-seed component passed; in-model seed, endpoints, and states pending"}

    def save() -> None:
        temporary = evidence / "results.tmp"
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
                             encoding="utf-8")
        temporary.replace(evidence / "results.json")

    save()
    try:
        source_spec = ct.models.utils.load_spec(str(source_package))
        tapped_source_spec = ct.models.utils.load_spec(str(tapped_package))
        correction_spec = ct.models.utils.load_spec(str(correction_package))
        report["source_preflight"] = inspect_specs(source_spec, tapped_source_spec, correction_spec, native)
        plain_spec, report["graph_check"] = rewrite(source_spec, correction_spec)
        tapped_spec, report["tapped_graph_check"] = add_taps(plain_spec, tapped_source_spec, source_spec)
        plain_destination = artifacts / "plain" / "0.mlpackage"
        tapped_destination = artifacts / "tapped" / "0.mlpackage"
        plain_destination.parent.mkdir(parents=True)
        tapped_destination.parent.mkdir(parents=True)
        shutil.copytree(source_package, plain_destination)
        shutil.copytree(source_package, tapped_destination)
        (plain_destination / MODEL_FILE).write_bytes(wire(plain_spec))
        (tapped_destination / MODEL_FILE).write_bytes(wire(tapped_spec))
        assert wire(ct.models.utils.load_spec(str(plain_destination))) == wire(plain_spec)
        assert wire(ct.models.utils.load_spec(str(tapped_destination))) == wire(tapped_spec)
        source_files = inventory(source_package)
        plain_files = inventory(plain_destination)
        tapped_files = inventory(tapped_destination)
        assert set(source_files) == set(plain_files) == set(tapped_files)
        assert [key for key in source_files if source_files[key] != plain_files[key]] == [MODEL_FILE]
        assert [key for key in plain_files if plain_files[key] != tapped_files[key]] == [MODEL_FILE]
        assert source_files["Data/com.apple.CoreML/weights/weight.bin"] == \
            plain_files["Data/com.apple.CoreML/weights/weight.bin"] == \
            tapped_files["Data/com.apple.CoreML/weights/weight.bin"]
        report.update(package=str(plain_destination.relative_to(ROOT)),
                      tapped_package=str(tapped_destination.relative_to(ROOT)),
                      source_files=source_files, variant_files=plain_files,
                      tapped_variant_files=tapped_files,
                      changed_package_files=[MODEL_FILE])
        save()
        if compile_models:
            from coreml_port.prediction_guard import install, stats

            report["guard_before_compile"] = install()
            assert stats()["plain_calls"] == stats()["state_calls"] == stats()["gil_releases"] == 0
            for key, path in (("compiled", plain_destination), ("tapped_compiled", tapped_destination)):
                compiled = path.parent / "0.mlmodelc"
                started = time.perf_counter()
                returned = ct.models.utils.compile_model(str(path), destination_path=str(compiled))
                assert Path(returned).resolve() == compiled.resolve()
                report[key] = str(compiled.relative_to(ROOT))
                report[key + "_files"] = inventory(compiled)
                report[key + "_compile_seconds"] = time.perf_counter() - started
                save()
            report["guard_after_compile"] = stats()
            assert report["guard_after_compile"] == report["guard_before_compile"]
        assert inventory(source_package) == source_files
        for path, expected in ((SOURCE_RESULT, SOURCE_RESULT_SHA), (TAPPED_RESULT, TAPPED_RESULT_SHA),
                               (HOST_RESULT, HOST_RESULT_SHA), (NATIVE_RESULT, NATIVE_RESULT_SHA),
                               (HELPER, HELPER_SHA)):
            assert sha(path) == expected
        assert sha(Path(__file__)) == report["script_sha256"]
        report.update(status="completed", passed=True,
                      scope="Structural package graft only. Native group seed/CPU preservation/GPU fidelity remain untested.")
    except BaseException as exc:
        report.update(status="failed", passed=False,
                      error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        save()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-preflight", action="store_true")
    parser.add_argument("--attempt")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    if args.source_preflight:
        if args.attempt or args.compile:
            parser.error("Source preflight cannot create or compile an attempt")
        result = source_preflight()
        path = ROOT / "evidence/06f-gpu-reciprocal/preflight.json"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite preflight: {path}")
        path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
                        encoding="utf-8")
        print(json.dumps({"status": result["status"], "passed": result["passed"],
                          "source_operation_count": result["source_graph_check"]["source_operation_count"]}))
    else:
        if not args.attempt:
            parser.error("--attempt required for build")
        result = build(args.attempt, args.compile)
        print(json.dumps({key: result[key] for key in
                          ("status", "passed", "package", "tapped_package", "seeded_component_passed")},
                         indent=2))


if __name__ == "__main__":
    main()
