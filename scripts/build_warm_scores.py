"""Isolated warm-only group-0 MIL ablation of unreachable tiny score trees.

The copied package keeps the qualified interface and every ordinary-score,
value-product, weight, and state operation. A separate harness must reject
anything except one real query at absolute position 32..2047.
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
SOURCE_AGGREGATE = "5a3e4c7a571e4fc3532377d9b9bb0da023f499c10709216fee7e1001bf4ad2c6"
PURE_SHORT_OPS = frozenset({
    "const", "add", "slice_by_index", "expand_dims", "mul", "select",
    "transpose", "equal", "shape", "concat", "minimum", "maximum",
    "reshape", "greater_equal", "gather", "sub", "real_div", "tile", "less",
})


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def wire(message) -> bytes:
    return message.SerializeToString(deterministic=True)


def inventory(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): sha(path)
            for path in sorted(root.rglob("*")) if path.is_file()}


def arguments(operation) -> list[str]:
    return [argument.name for binding in operation.inputs.values()
            for argument in binding.arguments if argument.name]


def one_input(operation, key: str) -> str:
    args = operation.inputs[key].arguments
    assert len(args) == 1 and args[0].name, (operation.type, key)
    return args[0].name


def operation_index(operations) -> dict[str, int]:
    names = [value.name for operation in operations for value in operation.outputs]
    assert len(names) == len(set(names)), "Duplicate MIL output name"
    return {value.name: index for index, operation in enumerate(operations)
            for value in operation.outputs}


def live_set(operations, producers: dict[str, int], outputs,
             selector_inputs: dict[int, str] | None = None) -> set[int]:
    roots = {producers[name] for name in outputs if name in producers}
    # Preserve all state operations and unknown zero-output effects regardless
    # of whether their outputs are needed by the two declared function outputs.
    roots.update(index for index, operation in enumerate(operations)
                 if not operation.outputs or "state" in operation.type)
    live: set[int] = set()
    pending = list(roots)
    while pending:
        index = pending.pop()
        if index in live:
            continue
        live.add(index)
        dependencies = ([selector_inputs[index]] if selector_inputs and index in selector_inputs
                        else arguments(operations[index]))
        pending.extend(producers[name] for name in dependencies if name in producers)
    return live


def upstream_of(names: list[str], operations, producers: dict[str, int]) -> set[int]:
    pending = [producers[name] for name in names if name in producers]
    upstream: set[int] = set()
    while pending:
        index = pending.pop()
        if index in upstream:
            continue
        upstream.add(index)
        pending.extend(producers[name] for name in arguments(operations[index])
                       if name in producers)
    return upstream


def check_graph(operations, outputs, external: set[str]) -> None:
    producers = operation_index(operations)
    assert set(outputs) <= set(producers) | external, "Missing function output producer"
    for index, operation in enumerate(operations):
        assert not operation.blocks, "Nested MIL block needs separate analysis"
        for name in arguments(operation):
            assert name in external or (name in producers and producers[name] < index), \
                f"Missing or forward MIL reference: {name} at {index}"


def build(attempt: str, compile_package: bool) -> dict:
    if not attempt or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in attempt):
        raise ValueError("Attempt must be a lowercase path-safe slug")
    evidence = ROOT / "evidence/05f-scores" / attempt
    artifact = ROOT / "artifacts/05f-scores" / attempt
    if evidence.exists() or artifact.exists():
        raise FileExistsError("Fresh evidence and artifact attempt paths required")
    evidence.mkdir(parents=True, exist_ok=False)
    artifact.mkdir(parents=True, exist_ok=False)
    (evidence / "source-snapshot.py").write_bytes(Path(__file__).read_bytes())
    report = {"status": "starting", "script_sha256": sha(Path(__file__)),
              "scope": "Harness-only one query, absolute position 32..2047; package interface unchanged; no runtime adoption"}

    def save() -> None:
        temporary = evidence / "results.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(evidence / "results.json")

    save()
    try:
        import coremltools as ct
        from coreml_port.package_identity import verify_candidate_identity

        source = ROOT / "artifacts/04d-cpu/full-candidate-6"
        source_identity = ROOT / "evidence/04d-cpu/full-candidate-6/package-identity.json"
        identity = verify_candidate_identity(source, source_identity)
        assert identity["aggregate_sha256"] == SOURCE_AGGREGATE
        report["source_candidate_identity"] = identity["aggregate_sha256"]
        src = source / "0.mlpackage"
        spec = ct.models.utils.load_spec(str(src))
        variant = copy.deepcopy(spec)
        assert set(spec.mlProgram.functions) == {"main"}
        assert set(spec.mlProgram.functions["main"].block_specializations) == {"CoreML8"}
        before = spec.mlProgram.functions["main"].block_specializations["CoreML8"]
        block = variant.mlProgram.functions["main"].block_specializations["CoreML8"]
        original = list(before.operations)
        operations = list(block.operations)
        assert len(original) == len(operations) == 42498
        assert not any(operation.blocks for operation in operations)
        producers = operation_index(operations)
        external = {name for operation in original for name in arguments(operation)
                    if name not in producers}
        assert external == {"x", "ve", "positions", "tap0", "tap12", "tap23",
                            "norm_divisor", *(f"inner_{kind}_cache_{layer}"
                                               for kind in ("k", "v") for layer in range(4))}
        check_graph(operations, block.outputs, external)

        selectors: dict[int, str] = {}
        short_roots: list[str] = []
        rewires: list[dict] = []
        for index, operation in enumerate(operations):
            if operation.type != "select" or len(operation.outputs) != 1:
                continue
            if not all(key in operation.inputs for key in ("a", "b", "cond")):
                continue
            short_name = one_input(operation, "a")
            if not short_name.startswith("short_scores"):
                continue
            full_name = one_input(operation, "b")
            condition_name = one_input(operation, "cond")
            assert operation.outputs[0].name in {"scores_1", "scores_3", "scores_5", "scores"}
            assert full_name in {"full_scores_1", "full_scores_3", "full_scores_5", "full_scores"}
            assert full_name.removeprefix("full_") == operation.outputs[0].name
            assert operations[producers[short_name]].type == "concat"
            assert operations[producers[full_name]].type == "mul"
            condition = operations[producers[condition_name]]
            assert condition.type == "less"
            threshold = operations[producers[one_input(condition, "y")]]
            assert threshold.type == "const"
            assert list(threshold.attributes["val"].immediateValue.tensor.ints.values) == [16]
            assert wire(operations[producers[full_name]].outputs[0].type) == wire(operation.outputs[0].type)
            selectors[index] = full_name
            short_roots.extend((short_name, condition_name))
            rewires.append({"selector": operation.outputs[0].name,
                            "ordinary_scores": full_name, "short_scores": short_name,
                            "condition": condition_name,
                            "ordinary_op_sha256": sha_bytes(wire(operations[producers[full_name]]))})
        assert len(selectors) == 4 and len({entry["selector"] for entry in rewires}) == 4

        live_before = live_set(operations, producers, block.outputs)
        live_after = live_set(operations, producers, block.outputs, selectors)
        assert live_after <= live_before
        removed = live_before - live_after
        short_upstream = upstream_of(short_roots, operations, producers)
        assert len(removed) == 10038 and removed <= short_upstream
        removed_types = Counter(operations[index].type for index in removed)
        assert set(removed_types) <= PURE_SHORT_OPS
        assert not any("state" in operations[index].type or not operations[index].outputs
                       for index in removed)
        assert all(index not in removed for index in selectors)
        for index, full_name in selectors.items():
            operation = operations[index]
            full_input = copy.deepcopy(operation.inputs["b"])
            operation.type = "identity"
            operation.ClearField("inputs")
            operation.inputs["x"].CopyFrom(full_input)
            assert one_input(operation, "x") == full_name
        survivors = [operation for index, operation in enumerate(operations) if index not in removed]
        del block.operations[:]
        block.operations.extend(survivors)
        check_graph(block.operations, block.outputs, external)
        survivor_original = [operation for index, operation in enumerate(original) if index not in removed]
        assert len(survivor_original) == len(block.operations)
        unchanged = 0
        for old, new in zip(survivor_original, block.operations):
            if old.outputs and old.outputs[0].name in {entry["selector"] for entry in rewires}:
                assert new.type == "identity" and wire(old.outputs[0]) == wire(new.outputs[0])
                assert wire(old.attributes["name"]) == wire(new.attributes["name"])
            else:
                assert wire(old) == wire(new), "Unrelated surviving operation changed"
                unchanged += 1
        before_shell = copy.deepcopy(spec)
        after_shell = copy.deepcopy(variant)
        del before_shell.mlProgram.functions["main"].block_specializations["CoreML8"].operations[:]
        del after_shell.mlProgram.functions["main"].block_specializations["CoreML8"].operations[:]
        assert wire(before_shell) == wire(after_shell), "Non-operation metadata changed"
        counts = lambda seq: dict(sorted(Counter(operation.type for operation in seq).items()))
        graph_check = {"passed": True, "before": counts(original), "after": counts(block.operations),
                       "selector_rewrites": rewires, "removed_operations": len(removed),
                       "removed_operation_types": dict(sorted(removed_types.items())),
                       "removed_output_names": sorted(value.name for index in removed
                                                      for value in original[index].outputs),
                       "unchanged_surviving_ops": unchanged,
                       "all_non_operation_metadata_identical": True,
                       "ordinary_matmuls_unchanged": counts(original).get("matmul") == counts(block.operations).get("matmul") == 68,
                       "state_reads_unchanged": counts(original).get("read_state") == counts(block.operations).get("read_state") == 16,
                       "state_writes_unchanged": counts(original).get("write_state") == counts(block.operations).get("write_state") == 8}
        assert graph_check["ordinary_matmuls_unchanged"] and graph_check["state_reads_unchanged"] and graph_check["state_writes_unchanged"]
        report["graph_check"] = graph_check
        report["rewires"] = rewires

        destination = artifact / "0.mlpackage"
        shutil.copytree(src, destination)
        spec_path = destination / "Data/com.apple.CoreML/model.mlmodel"
        spec_path.write_bytes(variant.SerializeToString())
        assert wire(ct.models.utils.load_spec(str(destination))) == wire(variant)
        source_files, variant_files = inventory(src), inventory(destination)
        changed = [name for name in source_files if source_files[name] != variant_files.get(name)]
        assert source_files.keys() == variant_files.keys()
        assert changed == ["Data/com.apple.CoreML/model.mlmodel"]
        report.update(package=str(destination.relative_to(ROOT)), source_files=source_files,
                      variant_files=variant_files, changed_package_files=changed)
        save()
        if compile_package:
            from coreml_port.prediction_guard import install
            install()
            started = time.perf_counter()
            compiled = artifact / "0.mlmodelc"
            returned = ct.models.utils.compile_model(str(destination), destination_path=str(compiled))
            assert Path(returned).resolve() == compiled.resolve()
            report.update(compile_seconds=time.perf_counter() - started,
                          compiled=str(compiled.relative_to(ROOT)),
                          compiled_files=inventory(compiled))
        assert sha(Path(__file__)) == report["script_sha256"] and inventory(src) == source_files
        report.update(status="completed", passed=True)
    except BaseException as exc:
        report.update(status="failed", passed=False,
                      error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        save()
    return report


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    result = build(args.attempt, args.compile)
    print(json.dumps({"passed": result["passed"],
                      "removed_operations": result["graph_check"]["removed_operations"],
                      "compiled": result.get("compiled"),
                      "compile_seconds": result.get("compile_seconds")}, indent=2))


if __name__ == "__main__":
    main()
