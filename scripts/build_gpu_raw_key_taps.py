"""Expose existing raw key projection and RoPE values in a group-0 clone.

This changes only the function and model-description output lists. It does not
rewrite any MIL operation, weight, input, state, or original endpoint. Native
endpoint/state controls under each compute policy are required before using
the additional outputs diagnostically.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE_AGGREGATE = "5a3e4c7a571e4fc3532377d9b9bb0da023f499c10709216fee7e1001bf4ad2c6"
TAPS = (("x_5", 535, "mul", (1, 1280)),
        ("x_9", 1141, "add", (1, 256)),
        ("value_5", 1169, "reshape", (1, 2, 128)),
        ("k_7", 1592, "mul", (1, 2, 128)),
        ("x_27", 1680, "mul", (1, 2, 128)),
        ("var_2083", 1706, "mul", (1, 2, 128)),
        ("x_29", 1713, "mul", (1, 2, 128)),
        ("var_2106", 1739, "mul", (1, 2, 128)),
        ("x_31", 1740, "add", (1, 2, 128)),
        ("step_23", 1763, "gather", (1, 2, 128)),
        ("var_2121", 1764, "real_div", (1, 2, 128)),
        ("var_2122", 1765, "round", (1, 2, 128)),
        ("k_9", 1766, "mul", (1, 2, 128)))
ORIGINAL_OUTPUTS = ("y", "tap23_out")
STATE_NAMES = tuple(f"inner_{kind}_cache_{layer}"
                    for layer in range(4) for kind in ("k", "v"))


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def wire(message) -> bytes:
    return message.SerializeToString(deterministic=True)


def inventory(root: Path) -> dict[str, dict[str, object]]:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Package contains symlink: {path}")
        if path.is_file():
            result[str(path.relative_to(root))] = {"size": path.stat().st_size,
                                                   "sha256": sha(path)}
        elif not path.is_dir():
            raise ValueError(f"Package contains nonregular entry: {path}")
    return result


def producer_index(operations) -> dict[str, int]:
    names = [value.name for operation in operations for value in operation.outputs]
    assert len(names) == len(set(names)), "Duplicate MIL value name"
    return {value.name: index for index, operation in enumerate(operations)
            for value in operation.outputs}


def one_input(operation, key: str) -> str:
    args = operation.inputs[key].arguments
    assert len(args) == 1 and args[0].name
    return args[0].name


def depends_on(value: str, ancestor: str, operations, producers: dict[str, int]) -> bool:
    pending = [value]
    seen: set[str] = set()
    while pending:
        name = pending.pop()
        if name == ancestor:
            return True
        if name in seen or name not in producers:
            continue
        seen.add(name)
        pending.extend(argument.name for binding in operations[producers[name]].inputs.values()
                       for argument in binding.arguments if argument.name)
    return False


def dimension_signature(mil_type) -> tuple[int | None, ...]:
    tensor = mil_type.tensorType
    assert tensor.dataType == 11  # MILProto DataType.FLOAT32
    values = []
    for dim in tensor.dimensions:
        if dim.HasField("constant"):
            values.append(dim.constant.size)
        else:
            assert dim.HasField("unknown")
            values.append(None)
    assert len(values) == tensor.rank
    return tuple(values)


def check_taps(operations) -> list[dict[str, object]]:
    producers = producer_index(operations)
    assert all(not operation.blocks for operation in operations)
    assert len(operations) > TAPS[-1][1]
    checks = []
    for name, index, op_type, fixed in TAPS:
        operation = operations[index]
        assert operation.type == op_type and producers[name] == index
        output = next(value for value in operation.outputs if value.name == name)
        shape = dimension_signature(output.type)
        assert shape[0] == fixed[0] and shape[1] is None and shape[2:] == fixed[1:]
        checks.append({"name": name, "operation_index": index,
                       "operation_type": op_type, "mil_shape": list(shape),
                       "operation_sha256": sha_bytes(wire(operation)),
                       "value_type_sha256": sha_bytes(wire(output.type))})
    assert all(depends_on(value, parent, operations, producers)
               for value, parent in (("x_9", "x_5"), ("value_5", "x_9"),
                                     ("k_7", "value_5"), ("x_27", "k_7"),
                                     ("var_2083", "x_27"), ("x_29", "k_7"),
                                     ("var_2106", "x_29"),
                                     ("x_31", "var_2083"), ("x_31", "var_2106"),
                                     ("step_23", "x_31"),
                                     ("var_2121", "step_23"), ("var_2121", "x_31"),
                                     ("var_2122", "var_2121"),
                                     ("k_9", "var_2122"), ("k_9", "step_23")))
    # These relationships pin the layer-0 path and the cached value being probed.
    assert one_input(operations[1790], "update") == "k_9"
    assert operations[1790].type == "slice_update"
    assert operations[1792].type == "cast" and one_input(operations[1792], "x") == \
        operations[1790].outputs[0].name
    assert operations[1793].type == "write_state" and \
        one_input(operations[1793], "data") == operations[1792].outputs[0].name and \
        one_input(operations[1793], "input") == "inner_k_cache_0"
    return checks


def append_output(description, name: str, dimensions: tuple[int | None, ...],
                  default_query_length: int, maximum_query_length: int) -> None:
    feature = description.output.add()
    feature.CopyFrom(description.output[0])
    feature.name = name
    array = feature.type.multiArrayType
    assert array.dataType == 65568  # ArrayFeatureType FLOAT32
    assert not array.shape and not array.shapeRange.sizeRanges
    for dim in dimensions:
        array.shape.append(default_query_length if dim is None else dim)
        bound = array.shapeRange.sizeRanges.add()
        bound.lowerBound = 1 if dim is None else dim
        bound.upperBound = maximum_query_length if dim is None else dim


def build(attempt: str, compile_package: bool) -> dict:
    if not attempt or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in attempt):
        raise ValueError("Attempt must be a lowercase path-safe slug")
    evidence = ROOT / "evidence/06c-gpu-localization" / attempt
    artifacts = ROOT / "artifacts/06c-gpu-localization" / attempt
    if evidence.exists() or artifacts.exists():
        raise FileExistsError("Fresh evidence and artifact attempt paths required")
    evidence.mkdir(parents=True, exist_ok=False)
    artifacts.mkdir(parents=True, exist_ok=False)
    (evidence / "source-snapshot.py").write_bytes(Path(__file__).read_bytes())
    design = ROOT / "evidence/06c-gpu-localization/raw-key-taps-design.md"
    (evidence / "design-snapshot.md").write_bytes(design.read_bytes())
    report = {"status": "starting", "passed": False,
              "script_sha256": sha(Path(__file__)), "design_sha256": sha(design),
              "source_candidate_identity": SOURCE_AGGREGATE,
              "scope": "Side-by-side group-0 raw key and RoPE taps; outputs only; diagnostic controls required"}

    def save() -> None:
        temp = evidence / "results.tmp"
        temp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temp.replace(evidence / "results.json")

    save()
    try:
        import coremltools as ct
        from coreml_port.package_identity import verify_candidate_identity

        source = ROOT / "artifacts/04d-cpu/full-candidate-6"
        manifest = ROOT / "evidence/04d-cpu/full-candidate-6/package-identity.json"
        identity = verify_candidate_identity(source, manifest)
        assert identity["aggregate_sha256"] == SOURCE_AGGREGATE
        src = source / "0.mlpackage"
        source_files = inventory(src)
        spec = ct.models.utils.load_spec(str(src))
        variant = copy.deepcopy(spec)
        assert set(spec.mlProgram.functions) == {"main"}
        assert set(spec.mlProgram.functions["main"].block_specializations) == {"CoreML8"}
        before = spec.mlProgram.functions["main"].block_specializations["CoreML8"]
        after = variant.mlProgram.functions["main"].block_specializations["CoreML8"]
        assert tuple(before.outputs) == ORIGINAL_OUTPUTS
        assert tuple(feature.name for feature in spec.description.output) == ORIGINAL_OUTPUTS
        assert tuple(feature.name for feature in spec.description.state) == STATE_NAMES
        assert len(before.operations) == len(after.operations)
        tap_checks = check_taps(before.operations)
        x_input = next(feature for feature in spec.description.input if feature.name == "x")
        x_array = x_input.type.multiArrayType
        assert list(x_array.shape) == [1, 3, 1280]
        x_ranges = x_array.shapeRange.sizeRanges
        assert len(x_ranges) == 3 and (x_ranges[1].lowerBound, x_ranges[1].upperBound) == (1, 32)
        default_query_length, maximum_query_length = x_array.shape[1], x_ranges[1].upperBound

        for tap in tap_checks:
            name = tap["name"]
            after.outputs.append(name)
            append_output(variant.description, name, tuple(tap["mil_shape"]),
                          default_query_length, maximum_query_length)
        assert tuple(after.outputs) == (*ORIGINAL_OUTPUTS, *(name for name, *_ in TAPS))
        assert tuple(feature.name for feature in variant.description.output) == tuple(after.outputs)
        original_operations_sha = sha_bytes(b"".join(wire(operation)
                                                   for operation in before.operations))
        variant_operations_sha = sha_bytes(b"".join(wire(operation)
                                                  for operation in after.operations))
        assert original_operations_sha == variant_operations_sha
        assert all(wire(old) == wire(new) for old, new in zip(before.operations, after.operations)), \
            "An original MIL operation changed"
        assert all(wire(old) == wire(new) for old, new in zip(spec.description.output,
                                                               variant.description.output[:len(ORIGINAL_OUTPUTS)]))
        before_shell, after_shell = copy.deepcopy(spec), copy.deepcopy(variant)
        del after_shell.description.output[len(ORIGINAL_OUTPUTS):]
        del after_shell.mlProgram.functions["main"].block_specializations["CoreML8"].outputs[len(ORIGINAL_OUTPUTS):]
        assert wire(before_shell) == wire(after_shell), \
            "An element outside the two output lists changed"
        output_schemas = [
            {"name": feature.name, "data_type": feature.type.multiArrayType.dataType,
             "default_shape": list(feature.type.multiArrayType.shape),
             "shape_ranges": [[bound.lowerBound, bound.upperBound]
                              for bound in feature.type.multiArrayType.shapeRange.sizeRanges]}
            for feature in variant.description.output[len(ORIGINAL_OUTPUTS):]
        ]
        graph_check = {"passed": True, "original_operations": len(before.operations),
                       "all_original_operations_byte_identical": True,
                       "original_operations_sha256": original_operations_sha,
                       "variant_operations_sha256": variant_operations_sha,
                       "only_description_and_block_outputs_changed": True,
                       "non_output_spec_sha256": sha_bytes(wire(before_shell)),
                       "original_outputs_preserved": list(ORIGINAL_OUTPUTS),
                       "state_names_preserved": list(STATE_NAMES),
                       "added_taps": tap_checks,
                       "output_schemas": output_schemas,
                       "default_query_length": default_query_length,
                       "query_length_range": [1, maximum_query_length]}
        report["graph_check"] = graph_check

        destination = artifacts / "0.mlpackage"
        shutil.copytree(src, destination)
        spec_path = destination / "Data/com.apple.CoreML/model.mlmodel"
        spec_path.write_bytes(variant.SerializeToString())
        assert wire(ct.models.utils.load_spec(str(destination))) == wire(variant)
        variant_files = inventory(destination)
        assert source_files.keys() == variant_files.keys()
        changed_files = [name for name in source_files
                         if source_files[name] != variant_files[name]]
        assert changed_files == ["Data/com.apple.CoreML/model.mlmodel"]
        assert all(source_files[name] == variant_files[name]
                   for name in source_files if name not in changed_files)
        report.update(package=str(destination.relative_to(ROOT)),
                      source_files=source_files, variant_files=variant_files,
                      changed_package_files=changed_files)
        save()
        if compile_package:
            from coreml_port.prediction_guard import install
            install()
            compiled = artifacts / "0.mlmodelc"
            started = time.perf_counter()
            returned = ct.models.utils.compile_model(str(destination),
                                                      destination_path=str(compiled))
            assert Path(returned).resolve() == compiled.resolve()
            report.update(compiled=str(compiled.relative_to(ROOT)),
                          compile_seconds=time.perf_counter() - started,
                          compiled_files=inventory(compiled))
        assert sha(Path(__file__)) == report["script_sha256"] and \
            sha(design) == report["design_sha256"] and inventory(src) == source_files
        report.update(status="completed", passed=True)
    except BaseException as exc:
        report.update(status="failed", passed=False,
                      error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        save()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    result = build(args.attempt, args.compile)
    print(json.dumps({"passed": result["passed"],
                      "added_taps": [tap["name"] for tap in result["graph_check"]["added_taps"]],
                      "compiled": result.get("compiled")}, indent=2))


if __name__ == "__main__":
    main()
