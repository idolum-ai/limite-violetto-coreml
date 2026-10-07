"""Expose twelve existing layer-0 normalization localization values in the round-fixed group-0 clone.

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
SOURCE_BUILD = ROOT / "evidence/06d-gpu-rounding/group0-1/results.json"
SOURCE_BUILD_SHA256 = "4e6ac1baa2f70c10bf812523f7e341d9dabf34b30299e4819631f2db31cf150f"
REPLACEMENT_PRODUCER_SHA256 = "b6d1baa5896f9f0dcd2b1ed9d95a01481f6326024a967acfff476711267b52ed"
TAPS = (("value_17", 8450, 9098, "mul", (1, None, 1280), "normalization_input"),
        ("x_65", 8451, 9099, "mul", (1, None, 1280), "square"),
        ("var_5669", 8888, 9536, "add", (1, None, 1), "ordered_sum"),
        ("mean_7", 8889, 9537, "real_div", (1, None, 1), "divide_by_norm_divisor"),
        ("var_5672", 8891, 9539, "add", (1, None, 1), "add_epsilon"),
        ("denom_7", 8892, 9540, "sqrt", (1, None, 1), "square_root"),
        ("var_5675", 8895, 9543, "real_div", (1, None, 1), "reciprocal"),
        ("x_67", 8896, 9544, "mul", (1, None, 1280), "raw_normalized_value"),
        ("step_47", 8919, 9567, "gather", (1, None, 1280), "quantization_step"),
        ("var_5690", 8920, 9568, "real_div", (1, None, 1280), "quantization_quotient"),
        ("var_5691", 8921, 9596, "select", (1, None, 1280), "replacement_round_result"),
        ("x_69", 8922, 9597, "mul", (1, None, 1280), "normalized_quantized_output"))
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


def inventory(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Package contains symlink: {path}")
        if path.is_file():
            result[str(path.relative_to(root))] = sha(path)
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


def check_taps(operations, baseline) -> list[dict[str, object]]:
    producers = producer_index(operations)
    baseline_producers = producer_index(baseline)
    assert len(operations) == 46035 and len(baseline) == 42498
    assert all(not op.blocks for op in operations)
    checks = []
    for name, original_index, index, op_type, expected_shape, stage in TAPS:
        operation = operations[index]
        assert operation.type == op_type and producers[name] == index
        assert baseline_producers[name] == original_index
        identical_to_baseline = name != "var_5691"
        if identical_to_baseline:
            assert wire(operation) == wire(baseline[original_index])
        else:
            assert baseline[original_index].type == "round"
            assert one_input(baseline[original_index], "x") == "var_5690"
            assert sha_bytes(wire(operation)) == REPLACEMENT_PRODUCER_SHA256
            assert one_input(operation, "a") == "var_5690"
            assert one_input(operation, "b") == "signed_even_site_8921_real_div_1"
            assert one_input(operation, "cond") == "signed_even_site_8921_equal_1"
            assert wire(operation.outputs[0]) == wire(baseline[original_index].outputs[0])
        output = next(value for value in operation.outputs if value.name == name)
        shape = dimension_signature(output.type)
        assert shape == expected_shape
        checks.append({"name": name, "stage": stage, "operation_index": index,
                       "original_operation_index": original_index,
                       "operation_type": op_type, "mil_shape": list(shape),
                       "operation_sha256": sha_bytes(wire(operation)),
                       "value_type_sha256": sha_bytes(wire(output.type)),
                       "producer_byte_identical_to_original": identical_to_baseline,
                       "direct_inputs": {key: [a.name for a in binding.arguments if a.name]
                                         for key, binding in operation.inputs.items()},
                       "candidate_replacement_producer_hash_pinned": name == "var_5691"})
    assert all(depends_on(value, parent, operations, producers)
               for value, parent in (("x_65", "value_17"), ("var_5669", "x_65"),
                                     ("mean_7", "var_5669"), ("var_5672", "mean_7"),
                                     ("denom_7", "var_5672"), ("var_5675", "denom_7"),
                                     ("x_67", "var_5675"), ("x_67", "value_17"),
                                     ("step_47", "x_67"), ("var_5690", "x_67"),
                                     ("var_5690", "step_47"), ("var_5691", "var_5690"),
                                     ("x_69", "var_5691"), ("x_69", "step_47")))
    expected_inputs = {"x_65": {"x": "value_17", "y": "value_17"},
                       "mean_7": {"x": "var_5669", "y": "norm_divisor"},
                       "var_5672": {"x": "mean_7", "y": "var_5671"},
                       "denom_7": {"x": "var_5672"},
                       "var_5675": {"x": "fill_like_4", "y": "denom_7"},
                       "x_67": {"x": "value_17", "y": "var_5675"},
                       "var_5690": {"x": "x_67", "y": "step_47"},
                       "x_69": {"x": "var_5691", "y": "step_47"}}
    for name, inputs in expected_inputs.items():
        op = operations[producers[name]]
        assert set(op.inputs) == set(inputs)
        assert all(one_input(op, key) == expected for key, expected in inputs.items())
    epsilon = operations[producers["var_5671"]]
    assert epsilon.type == "const"
    assert list(epsilon.attributes["val"].immediateValue.tensor.floats.values) == [2.0 ** -23]
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
    evidence = ROOT / "evidence/06e-gpu-normalization" / attempt
    artifacts = ROOT / "artifacts/06e-gpu-normalization" / attempt
    if evidence.exists() or artifacts.exists():
        raise FileExistsError("Fresh evidence and artifact attempt paths required")
    evidence.mkdir(parents=True, exist_ok=False)
    artifacts.mkdir(parents=True, exist_ok=False)
    (evidence / "source-snapshot.py").write_bytes(Path(__file__).read_bytes())
    design = ROOT / "evidence/06e-gpu-normalization/taps-design.md"
    (evidence / "design-snapshot.md").write_bytes(design.read_bytes())
    report = {"status": "starting", "passed": False,
              "script_sha256": sha(Path(__file__)), "design_sha256": sha(design),
              "source_candidate_identity": SOURCE_AGGREGATE,
              "scope": "Round-fixed group-0 twelve-tap layer-0 normalization localization; outputs only; same-policy endpoint/all-state byte controls required"}

    def save() -> None:
        temp = evidence / "results.tmp"
        temp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temp.replace(evidence / "results.json")

    save()
    try:
        import coremltools as ct
        assert sha(SOURCE_BUILD) == SOURCE_BUILD_SHA256
        source_build = json.loads(SOURCE_BUILD.read_text())
        assert source_build["status"] == "completed" and source_build["passed"] is True
        assert source_build["source_candidate_identity"] == SOURCE_AGGREGATE
        assert source_build["arithmetic_policy"] == "explicit-signed-ties-even-v3-round-only"
        assert source_build["graph_check"]["passed"] is True
        assert source_build["graph_check"]["replaced_round_count"] == 131
        assert source_build["graph_check"]["remaining_round_count"] == 0
        src = ROOT / source_build["package"]
        assert src == ROOT / "artifacts/06d-gpu-rounding/group0-1/0.mlpackage"
        source_files = inventory(src)
        assert source_files == source_build["variant_files"]
        baseline_path = ROOT / "artifacts/04d-cpu/full-candidate-6/0.mlpackage"
        baseline_model = baseline_path / "Data/com.apple.CoreML/model.mlmodel"
        assert sha(baseline_model) == source_build["source_files"]["Data/com.apple.CoreML/model.mlmodel"]
        baseline_spec = ct.models.utils.load_spec(str(baseline_path))
        baseline_ops = baseline_spec.mlProgram.functions["main"].block_specializations["CoreML8"].operations
        (evidence / "candidate-build-snapshot.json").write_bytes(SOURCE_BUILD.read_bytes())
        report.update(source_build_record=str(SOURCE_BUILD.relative_to(ROOT)),
                      source_build_sha256=SOURCE_BUILD_SHA256,
                      source_package=str(src.relative_to(ROOT)),
                      source_package_files=source_files,
                      source_arithmetic_policy=source_build["arithmetic_policy"],
                      native_instrumentation_control="pending: exact original endpoints and all8 states independently under both policies")
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
        tap_checks = check_taps(before.operations, baseline_ops)
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
                          default_query_length,
                          maximum_query_length)
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
            from coreml_port.prediction_guard import install, stats
            guard_before = install()
            report["guard_before_compile"] = guard_before
            compiled = artifacts / "0.mlmodelc"
            started = time.perf_counter()
            returned = ct.models.utils.compile_model(str(destination),
                                                      destination_path=str(compiled))
            assert Path(returned).resolve() == compiled.resolve()
            report.update(compiled=str(compiled.relative_to(ROOT)),
                          compile_seconds=time.perf_counter() - started,
                          compiled_files=inventory(compiled), guard_after_compile=stats())
            assert stats() == guard_before
        assert sha(Path(__file__)) == report["script_sha256"] and \
            sha(design) == report["design_sha256"] and inventory(src) == source_files
        assert sha(SOURCE_BUILD) == SOURCE_BUILD_SHA256
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
