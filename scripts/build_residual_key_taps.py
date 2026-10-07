"""Add output-only layer-2 key path taps to sealed 06f group 0.

No MIL operation, weight, state, input, or original endpoint is modified.
Native same-policy endpoint/state controls are required before interpreting taps.
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
SOURCE_BUILD = ROOT / "evidence/06f-gpu-reciprocal/group0-1/results.json"
SOURCE_BUILD_SHA256 = "48af7de89cde44bcaf7c7dda12457aec5eb9c2e99cfa1ae4c5294090529441d6"
SOURCE_MODEL_SHA256 = "dc0e08469999a5d2fc8b41d2a006861e886cf0d08ea28b31a4a3ba5fd2de2171"
SOURCE_AGGREGATE = "5a3e4c7a571e4fc3532377d9b9bb0da023f499c10709216fee7e1001bf4ad2c6"
SOURCE_OP_COUNT = 46185
ORIGINAL_OP_COUNT = 46035
INSERTED_OP_COUNT = 150
ORIGINAL_OUTPUTS = ("y", "tap23_out")
STATE_NAMES = tuple(f"inner_{kind}_cache_{layer}" for layer in range(4) for kind in ("k", "v"))
MODEL_FILE = "Data/com.apple.CoreML/model.mlmodel"
TAPS = (
    ("x_197", 24691, "add", (1, None, 256), "key_linear_raw"),
    ("value_41", 24746, "reshape", (1, None, 2, 128), "key_projection_quantized"),
    ("denom_21", 25191, "sqrt", (1, None, 2, 1), "key_norm_denominator"),
    ("var_14324", 25194, "real_div", (1, None, 2, 1), "key_norm_reciprocal_seed"),
    ("x_207", 25195, "mul", (1, None, 2, 128), "key_norm_raw"),
    ("step_149", 25218, "gather", (1, None, 2, 128), "key_norm_quantization_step"),
    ("var_14339", 25219, "real_div", (1, None, 2, 128), "key_norm_quotient"),
    ("var_14340", 25247, "select", (1, None, 2, 128), "key_norm_signed_even_round"),
    ("k_49", 25248, "mul", (1, None, 2, 128), "key_norm_quantized_rope_input"),
    ("x_215", 25417, "mul", (1, None, 2, 128), "rope_cosine_product_raw"),
    ("var_14415", 25470, "mul", (1, None, 2, 128), "rope_cosine_product_quantized"),
    ("x_217", 25477, "mul", (1, None, 2, 128), "rope_sine_product_raw"),
    ("var_14438", 25530, "mul", (1, None, 2, 128), "rope_sine_product_quantized"),
    ("x_219", 25531, "add", (1, None, 2, 128), "rope_sum_raw"),
    ("step_161", 25554, "gather", (1, None, 2, 128), "final_key_quantization_step"),
    ("var_14453", 25555, "real_div", (1, None, 2, 128), "final_key_quotient"),
    ("var_14454", 25583, "select", (1, None, 2, 128), "final_key_signed_even_round"),
    ("k_51", 25584, "mul", (1, None, 2, 128), "final_key_before_cache"),
)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def wire(message) -> bytes:
    return message.SerializeToString(deterministic=True)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def inventory(root: Path) -> dict[str, str]:
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlink in package: {path}")
        if path.is_file():
            files[str(path.relative_to(root))] = sha(path)
        elif not path.is_dir():
            raise ValueError(f"Nonregular package entry: {path}")
    return files


def block(spec):
    assert set(spec.mlProgram.functions) == {"main"}
    function = spec.mlProgram.functions["main"]
    assert set(function.block_specializations) == {"CoreML8"}
    return function.block_specializations["CoreML8"]


def one_input(op, key: str) -> str:
    arguments = op.inputs[key].arguments
    assert len(arguments) == 1 and arguments[0].name
    return arguments[0].name


def depends_on(value: str, ancestor: str, operations, producers: dict[str, int]) -> bool:
    pending = [value]
    seen = set()
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


def dims(value) -> tuple[int | None, ...]:
    tensor = value.type.tensorType
    assert tensor.dataType == 11  # FLOAT32
    result = []
    for dim in tensor.dimensions:
        if dim.HasField("constant"):
            result.append(dim.constant.size)
        else:
            assert dim.HasField("unknown")
            result.append(None)
    assert len(result) == tensor.rank
    return tuple(result)


def context():
    assert sha(SOURCE_BUILD) == SOURCE_BUILD_SHA256
    record = json.loads(SOURCE_BUILD.read_text(encoding="utf-8"))
    assert record["status"] == "completed" and record["passed"] is True
    assert record["source_candidate_identity"] == SOURCE_AGGREGATE
    assert record["graph_check"]["passed"] and record["tapped_graph_check"]["passed"]
    assert record["package"] == "artifacts/06f-gpu-reciprocal/group0-1/plain/0.mlpackage"
    assert record["variant_files"][MODEL_FILE] == SOURCE_MODEL_SHA256
    source = ROOT / record["package"]
    source_files = inventory(source)
    assert source_files == record["variant_files"]
    assert record["source_files"] == json.loads(
        (ROOT / "evidence/06d-gpu-rounding/group0-1/results.json").read_text()
    )["variant_files"]
    return record, source, source_files


def inspect(spec, source_files: dict[str, str]) -> tuple[object, dict]:
    ops = list(block(spec).operations)
    assert len(ops) == SOURCE_OP_COUNT and all(not op.blocks for op in ops)
    assert tuple(block(spec).outputs) == ORIGINAL_OUTPUTS
    assert tuple(feature.name for feature in spec.description.output) == ORIGINAL_OUTPUTS
    assert tuple(feature.name for feature in spec.description.state) == STATE_NAMES
    names = [value.name for op in ops for value in op.outputs]
    assert len(names) == len(set(names))
    producers = {value.name: i for i, op in enumerate(ops) for value in op.outputs}
    checks = []
    for name, index, kind, expected_shape, stage in TAPS:
        assert producers[name] == index
        op = ops[index]
        assert op.type == kind and len(op.outputs) == 1
        assert dims(op.outputs[0]) == expected_shape
        checks.append({"name": name, "stage": stage, "operation_index": index,
                       "original_operation_index": index - INSERTED_OP_COUNT,
                       "operation_type": kind, "mil_shape": list(expected_shape),
                       "operation_sha256": digest(wire(op)),
                       "direct_inputs": {key: [arg.name for arg in binding.arguments if arg.name]
                                         for key, binding in op.inputs.items()}})
    assert one_input(ops[producers["value_41"]], "x") == "var_13956"
    assert one_input(ops[producers["denom_21"]], "x") == "var_14321"
    reciprocal = ops[producers["var_14324"]]
    assert one_input(reciprocal, "y") == "denom_21"
    filled = one_input(reciprocal, "x")
    assert filled == "fill_like_12"
    fill = ops[producers[filled]]
    assert fill.type == "fill_like" and one_input(fill, "ref_tensor") == "denom_21"
    fill_const = ops[producers[one_input(fill, "value")]]
    assert fill_const.type == "const"
    assert list(fill_const.attributes["val"].immediateValue.tensor.floats.values) == [1.0]
    for name, inputs in {
        "x_207": {"x": "value_41", "y": "var_14324"},
        "var_14339": {"x": "x_207", "y": "step_149"},
        "k_49": {"x": "var_14340", "y": "step_149"},
        "x_215": {"x": "cos_1", "y": "k_49"},
        "x_217": {"x": "sin_1", "y": "var_14421"},
        "x_219": {"x": "var_14415", "y": "var_14438"},
        "var_14453": {"x": "x_219", "y": "step_161"},
        "k_51": {"x": "var_14454", "y": "step_161"},
    }.items():
        op = ops[producers[name]]
        assert set(op.inputs) == set(inputs)
        assert all(one_input(op, key) == expected for key, expected in inputs.items())
    assert all(depends_on(value, ancestor, ops, producers) for value, ancestor in (
        ("x_197", "x_193"), ("value_41", "x_197"),
        ("denom_21", "value_41"), ("var_14324", "denom_21"),
        ("x_207", "value_41"), ("x_207", "var_14324"),
        ("var_14339", "x_207"), ("k_49", "var_14339"),
        ("x_215", "k_49"), ("var_14415", "x_215"),
        ("x_217", "k_49"), ("var_14438", "x_217"),
        ("x_219", "var_14415"), ("x_219", "var_14438"),
        ("step_161", "x_219"), ("var_14453", "x_219"),
        ("var_14453", "step_161"), ("var_14454", "var_14453"),
        ("k_51", "var_14454")))
    assert producers["var_14324"] == 25044 + INSERTED_OP_COUNT
    assert ops[25592].type == "slice_update" and one_input(ops[25592], "update") == "k_51"
    assert source_files[MODEL_FILE] == SOURCE_MODEL_SHA256
    variant = copy.deepcopy(spec)
    query = next(feature for feature in spec.description.input if feature.name == "x")
    array = query.type.multiArrayType
    assert list(array.shape) == [1, 3, 1280]
    ranges = array.shapeRange.sizeRanges
    assert len(ranges) == 3 and (ranges[1].lowerBound, ranges[1].upperBound) == (1, 32)
    default_length, max_length = array.shape[1], ranges[1].upperBound
    for tap in checks:
        block(variant).outputs.append(tap["name"])
        feature = variant.description.output.add()
        feature.CopyFrom(spec.description.output[0])
        feature.name = tap["name"]
        output_array = feature.type.multiArrayType
        assert output_array.dataType == 65568 and not output_array.shape and not output_array.shapeRange.sizeRanges
        for dimension in tap["mil_shape"]:
            output_array.shape.append(default_length if dimension is None else dimension)
            bound = output_array.shapeRange.sizeRanges.add()
            bound.lowerBound = 1 if dimension is None else dimension
            bound.upperBound = max_length if dimension is None else dimension
    assert tuple(block(variant).outputs) == (*ORIGINAL_OUTPUTS, *(tap[0] for tap in TAPS))
    assert tuple(feature.name for feature in variant.description.output) == tuple(block(variant).outputs)
    before_ops = digest(b"".join(wire(op) for op in block(spec).operations))
    after_ops = digest(b"".join(wire(op) for op in block(variant).operations))
    assert before_ops == after_ops
    assert all(wire(a) == wire(b) for a, b in zip(block(spec).operations, block(variant).operations))
    shell = copy.deepcopy(variant)
    del shell.description.output[len(ORIGINAL_OUTPUTS):]
    del block(shell).outputs[len(ORIGINAL_OUTPUTS):]
    assert wire(shell) == wire(spec)
    check = {"passed": True, "source_operation_count": len(ops),
             "all_operations_byte_identical": True, "operations_sha256": before_ops,
             "only_description_and_block_outputs_changed": True,
             "key_dependency_chain_valid": True,
             "original_endpoints_preserved": list(ORIGINAL_OUTPUTS),
             "state_names_preserved": list(STATE_NAMES),
             "added_taps": checks, "default_query_length": default_length,
             "query_length_range": [1, max_length],
             "non_output_spec_sha256": digest(wire(shell))}
    return variant, check


def preflight() -> dict:
    import coremltools as ct
    record, source, source_files = context()
    spec = ct.models.utils.load_spec(str(source))
    _variant, check = inspect(spec, source_files)
    return {"status": "completed", "passed": True,
            "script_sha256": sha(Path(__file__)),
            "source_build_sha256": SOURCE_BUILD_SHA256,
            "source_candidate_identity": SOURCE_AGGREGATE,
            "source_package": record["package"], "source_files": source_files,
            "graph_check": check,
            "scope": "Static saved-spec output-only tap rewrite; no package build or native calls"}


def build(attempt: str, compile_package: bool) -> dict:
    import coremltools as ct
    if not attempt or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in attempt):
        raise ValueError("A fresh path-safe attempt is required")
    evidence = ROOT / "evidence/06g-gpu-residual" / attempt
    artifacts = ROOT / "artifacts/06g-gpu-residual" / attempt
    if evidence.exists() or artifacts.exists():
        raise FileExistsError("Fresh evidence/artifact attempt required")
    record, source, source_files = context()
    evidence.mkdir(parents=True)
    artifacts.mkdir(parents=True)
    design = ROOT / "evidence/06g-gpu-residual/key-design.md"
    for name, path in (("source-snapshot.py", Path(__file__)),
                       ("source-build-snapshot.json", SOURCE_BUILD),
                       ("design-snapshot.md", design)):
        (evidence / name).write_bytes(path.read_bytes())
    report = {"status": "starting", "passed": False, "attempt": attempt,
              "script_sha256": sha(Path(__file__)), "design_sha256": sha(design),
              "source_build_sha256": SOURCE_BUILD_SHA256,
              "source_candidate_identity": SOURCE_AGGREGATE,
              "source_package": record["package"], "source_files": source_files,
              "native_endpoint_state_control": "pending: same-policy original-versus-tapped outputs and all 8 states"}

    def save():
        path = evidence / "results.tmp"
        path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        path.replace(evidence / "results.json")

    save()
    try:
        spec = ct.models.utils.load_spec(str(source))
        variant, report["graph_check"] = inspect(spec, source_files)
        destination = artifacts / "0.mlpackage"
        shutil.copytree(source, destination)
        (destination / MODEL_FILE).write_bytes(wire(variant))
        assert wire(ct.models.utils.load_spec(str(destination))) == wire(variant)
        files = inventory(destination)
        assert set(source_files) == set(files)
        assert [name for name in source_files if source_files[name] != files[name]] == [MODEL_FILE]
        assert source_files["Data/com.apple.CoreML/weights/weight.bin"] == \
            files["Data/com.apple.CoreML/weights/weight.bin"]
        report.update(package=str(destination.relative_to(ROOT)), variant_files=files,
                      changed_package_files=[MODEL_FILE])
        save()
        if compile_package:
            from coreml_port.prediction_guard import install, stats
            report["guard_before_compile"] = install()
            assert stats()["plain_calls"] == stats()["state_calls"] == stats()["gil_releases"] == 0
            compiled = artifacts / "0.mlmodelc"
            started = time.perf_counter()
            returned = ct.models.utils.compile_model(str(destination), destination_path=str(compiled))
            assert Path(returned).resolve() == compiled.resolve()
            report.update(compiled=str(compiled.relative_to(ROOT)),
                          compiled_files=inventory(compiled),
                          compile_seconds=time.perf_counter() - started,
                          guard_after_compile=stats())
            assert report["guard_after_compile"] == report["guard_before_compile"]
        assert sha(Path(__file__)) == report["script_sha256"] and sha(design) == report["design_sha256"]
        assert sha(SOURCE_BUILD) == SOURCE_BUILD_SHA256 and inventory(source) == source_files
        assert inventory(destination) == files
        report.update(status="completed", passed=True,
                      scope="Output-only diagnostic clone; native controls pending")
    except BaseException as exc:
        report.update(status="failed", passed=False,
                      error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--attempt")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        if args.attempt or args.compile:
            parser.error("--preflight does not build or compile")
        report = preflight()
        path = ROOT / "evidence/06g-gpu-residual/key-preflight.json"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        print(json.dumps({"passed": report["passed"],
                          "taps": [tap["name"] for tap in report["graph_check"]["added_taps"]]}))
    else:
        if not args.attempt:
            parser.error("--attempt is required for a build")
        report = build(args.attempt, args.compile)
        print(json.dumps({"passed": report["passed"], "package": report["package"],
                          "compiled": report.get("compiled")}, indent=2))


if __name__ == "__main__":
    main()
