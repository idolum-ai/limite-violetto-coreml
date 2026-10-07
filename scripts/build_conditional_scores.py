"""Build a side-by-side Core ML group with lazy short-score conditionals.

The qualified candidate remains untouched. Four eager score selects become
cond(pred=rows<16): true branches contain their original tiny score trees;
false branches return the already-computed ordinary score tensor. No state
operation moves, and all original arithmetic operations remain byte-identical.
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
    "const", "add", "slice_by_index", "expand_dims", "mul", "select", "transpose",
    "equal", "shape", "concat", "minimum", "maximum", "reshape", "greater_equal",
    "gather", "sub", "real_div", "tile", "less",
})
EXTERNALS = {"x", "ve", "positions", "tap0", "tap12", "tap23", "norm_divisor",
             *(f"inner_{kind}_cache_{layer}" for kind in ("k", "v") for layer in range(4))}


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def producer_index(operations) -> dict[str, int]:
    names = [value.name for operation in operations for value in operation.outputs]
    assert len(names) == len(set(names)), "Duplicate MIL output name"
    return {value.name: index for index, operation in enumerate(operations)
            for value in operation.outputs}


def closure(start: set[int], operations, producers: dict[str, int],
            stop: set[int] = frozenset()) -> set[int]:
    found: set[int] = set()
    pending = list(start)
    while pending:
        index = pending.pop()
        if index in found:
            continue
        found.add(index)
        if index not in stop:
            pending.extend(producers[name] for name in arguments(operations[index])
                           if name in producers)
    return found


def live_set(operations, producers: dict[str, int], outputs,
             selectors: dict[int, tuple[str, str, str]] | None = None) -> set[int]:
    roots = {producers[name] for name in outputs if name in producers}
    roots.update(index for index, operation in enumerate(operations)
                 if not operation.outputs or "state" in operation.type)
    found: set[int] = set()
    pending = list(roots)
    while pending:
        index = pending.pop()
        if index in found:
            continue
        found.add(index)
        names = ([selectors[index][1], selectors[index][2]]
                 if selectors and index in selectors else arguments(operations[index]))
        pending.extend(producers[name] for name in names if name in producers)
    return found


def check_parent(operations, outputs) -> None:
    producers = producer_index(operations)
    assert set(outputs) <= set(producers) | EXTERNALS
    for index, operation in enumerate(operations):
        for name in arguments(operation):
            assert name in EXTERNALS or (name in producers and producers[name] < index), \
                f"Parent MIL reference unavailable: {name} at {index}"


def locate_selectors(operations, producers) -> dict[int, tuple[str, str, str]]:
    selected = {}
    for index, operation in enumerate(operations):
        if operation.type != "select" or len(operation.outputs) != 1 or \
                not all(key in operation.inputs for key in ("a", "b", "cond")):
            continue
        short_name = one_input(operation, "a")
        if not short_name.startswith("short_scores"):
            continue
        full_name = one_input(operation, "b")
        predicate_name = one_input(operation, "cond")
        assert full_name.startswith("full_scores")
        assert full_name.removeprefix("full_") == operation.outputs[0].name
        assert operations[producers[short_name]].type == "concat"
        assert operations[producers[full_name]].type == "mul"
        predicate = operations[producers[predicate_name]]
        assert predicate.type == "less" and predicate.outputs[0].type.tensorType.rank == 0
        threshold = operations[producers[one_input(predicate, "y")]]
        assert threshold.type == "const" and \
            list(threshold.attributes["val"].immediateValue.tensor.ints.values) == [16]
        short_type = operations[producers[short_name]].outputs[0].type.tensorType
        full_type = operations[producers[full_name]].outputs[0].type.tensorType
        assert wire(operations[producers[full_name]].outputs[0].type) == wire(operation.outputs[0].type)
        assert short_type.dataType == full_type.dataType and short_type.rank == full_type.rank == 4
        assert wire(short_type.dimensions[0]) == wire(full_type.dimensions[0])
        assert wire(short_type.dimensions[1]) == wire(full_type.dimensions[1])
        assert short_type.dimensions[3].HasField("unknown") and \
            full_type.dimensions[3].constant.size == 2048
        selected[index] = (short_name, full_name, predicate_name)
    assert len(selected) == 4, f"Expected four tiny/full score selectors, got {len(selected)}"
    return selected


def partition_short_branches(operations, producers, outputs, selectors):
    live_before = live_set(operations, producers, outputs)
    live_keep = live_set(operations, producers, outputs, selectors)
    assert live_keep <= live_before
    short_only = live_before - live_keep
    assert short_only and all(operations[index].type in PURE_SHORT_OPS and
                              operations[index].outputs and "state" not in operations[index].type
                              for index in short_only)
    ancestors = {selector: closure({producers[names[0]]}, operations, producers,
                                   stop=set(selectors))
                 for selector, names in selectors.items()}
    shared = {index for index in short_only
              if sum(index in upstream for upstream in ancestors.values()) > 1}
    # Shared ancestors stay in the parent, along with their dependencies.
    shared_kept = closure(shared, operations, producers, stop=set(selectors)) & short_only
    branches = {selector: (upstream & short_only) - shared_kept
                for selector, upstream in ancestors.items()}
    assigned = set().union(*branches.values(), shared_kept)
    assert assigned == short_only
    assert all(branches[selector] and producers[selectors[selector][0]] in branches[selector]
               for selector in selectors)
    assert all(not branches[a] & branches[b] for a in branches for b in branches if a != b)
    # An output moved to a branch may be used only within that branch or as
    # its result. Any shared/other consumer must remain in parent instead.
    for selector, branch in branches.items():
        for index, operation in enumerate(operations):
            if index in branch:
                continue
            names = ([selectors[index][1], selectors[index][2]]
                     if index in selectors else arguments(operation))
            assert all(producers.get(name) not in branch for name in names), \
                f"Moved value still consumed outside true block: selector={selector} op={index}"
        for other, other_branch in branches.items():
            if other != selector:
                assert all(producers.get(name) not in branch for index in other_branch
                           for name in arguments(operations[index]))
    return short_only, shared_kept, branches


def check_nested(operations, branches, selectors, original, external_parent) -> None:
    parent_producers = producer_index(operations)
    assert external_parent == EXTERNALS
    for parent_index, operation in enumerate(operations):
        if operation.type != "cond":
            assert not operation.blocks
            continue
        assert len(operation.blocks) == 2 and set(operation.inputs) == {"pred"}
        predicate_name = one_input(operation, "pred")
        assert predicate_name in parent_producers and parent_producers[predicate_name] < parent_index
        true_block, false_block = operation.blocks
        assert len(true_block.outputs) == len(false_block.outputs) == 1
        assert not true_block.inputs and not false_block.inputs
        true_producers = producer_index(true_block.operations)
        false_producers = producer_index(false_block.operations)
        assert true_block.outputs[0] in true_producers
        assert len(false_block.operations) == 1 and false_block.outputs[0] in false_producers
        false_identity = false_block.operations[0]
        assert false_identity.type == "identity" and not false_identity.blocks
        false_input = one_input(false_identity, "x")
        assert false_input in parent_producers and parent_producers[false_input] < parent_index
        for local_index, child in enumerate(true_block.operations):
            assert not child.blocks and child.type in PURE_SHORT_OPS | {"identity"}
            for name in arguments(child):
                if name in true_producers:
                    assert true_producers[name] < local_index
                else:
                    assert name in EXTERNALS or (name in parent_producers and
                                                parent_producers[name] < parent_index), \
                        f"Branch capture unavailable: {name}"


def build(group: str, attempt: str, compile_package: bool) -> dict:
    if group not in {str(index) for index in range(12)}:
        raise ValueError("Group must be 0..11")
    if not attempt or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in attempt):
        raise ValueError("Attempt must be a lowercase path-safe slug")
    evidence = ROOT / "evidence/05g-integration" / attempt
    artifact = ROOT / "artifacts/05g-integration" / attempt
    if evidence.exists() or artifact.exists():
        raise FileExistsError("Fresh attempt paths required")
    evidence.mkdir(parents=True, exist_ok=False)
    artifact.mkdir(parents=True, exist_ok=False)
    (evidence / "source-snapshot.py").write_bytes(Path(__file__).read_bytes())
    report = {"status": "starting", "group": group, "script_sha256": sha(Path(__file__)),
              "scope": "Single stateful group; lazy Q<16 tiny-score branch, ordinary score branch otherwise; all original input shapes"}

    def save() -> None:
        temporary = evidence / "results.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(evidence / "results.json")

    save()
    try:
        import coremltools as ct
        from coreml_port.package_identity import verify_candidate_identity

        source = ROOT / "artifacts/04d-cpu/full-candidate-6"
        identity = verify_candidate_identity(
            source, ROOT / "evidence/04d-cpu/full-candidate-6/package-identity.json")
        assert identity["aggregate_sha256"] == SOURCE_AGGREGATE
        report["source_candidate_identity"] = identity["aggregate_sha256"]
        src = source / f"{group}.mlpackage"
        spec = ct.models.utils.load_spec(str(src))
        variant = copy.deepcopy(spec)
        assert set(spec.mlProgram.functions) == {"main"}
        assert set(spec.mlProgram.functions["main"].block_specializations) == {"CoreML8"}
        before = spec.mlProgram.functions["main"].block_specializations["CoreML8"]
        block = variant.mlProgram.functions["main"].block_specializations["CoreML8"]
        original, operations = list(before.operations), list(block.operations)
        assert len(original) == len(operations) and not any(operation.blocks for operation in operations)
        producers = producer_index(operations)
        external = {name for operation in operations for name in arguments(operation)
                    if name not in producers}
        check_parent(operations, block.outputs)
        assert external == EXTERNALS
        selectors = locate_selectors(operations, producers)
        short_only, shared_kept, branches = partition_short_branches(
            operations, producers, block.outputs, selectors)
        identity_template = next((operation for operation in original
                                  if operation.type == "identity"), None)
        if identity_template is None:
            # Some qualified groups have no original identity. The source
            # aggregate above also covers group 0, whose identity is a
            # canonical MIL protobuf template for the same target program.
            template_spec = ct.models.utils.load_spec(str(source / "0.mlpackage"))
            template_ops = template_spec.mlProgram.functions["main"].block_specializations["CoreML8"].operations
            identity_template = next(operation for operation in template_ops
                                     if operation.type == "identity")
        assert set(identity_template.inputs) == {"x"} and len(identity_template.outputs) == 1

        selector_records = []
        for index, names in selectors.items():
            short_name, full_name, predicate_name = names
            old = original[index]
            operation = operations[index]
            predicate_input = copy.deepcopy(operation.inputs["cond"])
            operation.type = "cond"
            operation.ClearField("inputs")
            operation.inputs["pred"].CopyFrom(predicate_input)
            true_block = operation.blocks.add()
            true_block.operations.extend(operations[child] for child in sorted(branches[index]))
            refined_name = f"{short_name}_conditional_refined"
            assert refined_name not in producers
            refinement = copy.deepcopy(identity_template)
            refinement.inputs["x"].arguments[0].name = short_name
            refinement.outputs[0].name = refined_name
            refinement.outputs[0].type.CopyFrom(old.outputs[0].type)
            refinement.attributes["name"].immediateValue.tensor.strings.values[0] = refined_name
            true_block.operations.append(refinement)
            true_block.outputs.append(refined_name)
            false_block = operation.blocks.add()
            false_name = f"{full_name}_conditional_return"
            assert false_name not in producers
            false_return = copy.deepcopy(identity_template)
            false_return.inputs["x"].arguments[0].name = full_name
            false_return.outputs[0].name = false_name
            false_return.outputs[0].type.CopyFrom(old.outputs[0].type)
            false_return.attributes["name"].immediateValue.tensor.strings.values[0] = false_name
            false_block.operations.append(false_return)
            false_block.outputs.append(false_name)
            assert wire(operation.outputs[0]) == wire(old.outputs[0])
            assert wire(operation.attributes["name"]) == wire(old.attributes["name"])
            selector_records.append({"output": operation.outputs[0].name,
                                     "predicate": predicate_name,
                                     "predicate_rule": "prepared_query_rows < 16",
                                     "threshold": 16,
                                     "short_scores": short_name, "ordinary_scores": full_name,
                                     "short_type_sha256": sha_bytes(wire(original[producers[short_name]].outputs[0].type)),
                                     "ordinary_type_sha256": sha_bytes(wire(original[producers[full_name]].outputs[0].type)),
                                     "output_type_sha256": sha_bytes(wire(operation.outputs[0].type)),
                                     "true_block_ops": len(branches[index]) + 1,
                                     "shape_refinement": refined_name,
                                     "false_block_output": false_name,
                                     "true_block_sha256": sha_bytes(b"".join(
                                         wire(original[child]) for child in sorted(branches[index]))),
                                     "false_block_ops": 1})
        moved = set().union(*branches.values())
        survivors = [operation for index, operation in enumerate(operations) if index not in moved]
        del block.operations[:]
        block.operations.extend(survivors)
        check_parent(block.operations, block.outputs)
        check_nested(block.operations, branches, selectors, original, external)

        original_survivors = [operation for index, operation in enumerate(original) if index not in moved]
        assert len(original_survivors) == len(block.operations)
        changed = 0
        unchanged = 0
        for old, new in zip(original_survivors, block.operations):
            if old.type == "select" and new.type == "cond":
                changed += 1
            else:
                assert wire(old) == wire(new), "Unrelated parent operation changed"
                unchanged += 1
        assert changed == 4
        for selector, branch in branches.items():
            nested = operations[selector].blocks[0]
            assert len(nested.operations) == len(branch) + 1
            for child, index in zip(nested.operations[:-1], sorted(branch)):
                assert wire(child) == wire(original[index]), "Moved short operation changed"
            assert nested.operations[-1].type == "identity" and \
                wire(nested.operations[-1].outputs[0].type) == wire(original[selector].outputs[0].type)
            false_nested = operations[selector].blocks[1]
            assert len(false_nested.operations) == 1 and \
                false_nested.operations[0].type == "identity" and \
                one_input(false_nested.operations[0], "x") == selectors[selector][1] and \
                wire(false_nested.operations[0].outputs[0].type) == \
                wire(original[selector].outputs[0].type)
        assert unchanged + changed + len(moved) == len(original)
        shell_before, shell_after = copy.deepcopy(spec), copy.deepcopy(variant)
        del shell_before.mlProgram.functions["main"].block_specializations["CoreML8"].operations[:]
        del shell_after.mlProgram.functions["main"].block_specializations["CoreML8"].operations[:]
        assert wire(shell_before) == wire(shell_after), "Non-operation metadata changed"
        counts = lambda seq: dict(sorted(Counter(operation.type for operation in seq).items()))
        original_counts = counts(original)
        after_all = counts([operation for parent in block.operations
                            for operation in (parent, *[child for nested in parent.blocks
                                                     for child in nested.operations])])
        assert after_all.get("identity") == original_counts.get("identity", 0) + 8
        assert original_counts.get("matmul", 0) > 0 and \
            original_counts.get("matmul") == after_all.get("matmul")
        assert original_counts.get("read_state", 0) > 0 and \
            original_counts.get("read_state") == after_all.get("read_state")
        assert original_counts.get("write_state", 0) > 0 and \
            original_counts.get("write_state") == after_all.get("write_state")
        assert len(short_only) == len(moved) + len(shared_kept)
        graph_check = {"passed": True, "before": original_counts, "after_all_blocks": after_all,
                       "selectors": selector_records, "selector_rewrites": 4,
                       "exclusive_short_ops": len(short_only), "moved_short_ops": len(moved),
                       "shared_short_ops_kept_outside": len(shared_kept),
                       "true_block_ops": [len(branches[index]) + 1 for index in selectors],
                       "new_branch_shape_refinement_identities": 4,
                       "new_false_branch_return_identities": 4,
                       "unchanged_surviving_ops": unchanged,
                       "all_moved_ops_byte_identical": True,
                       "all_non_operation_metadata_identical": True,
                       "value_product_matmuls_unchanged": True,
                       "state_reads_unchanged": True, "state_writes_unchanged": True,
                       "single_model_state_interface_preserved": True}
        report["graph_check"] = graph_check

        destination = artifact / f"{group}.mlpackage"
        shutil.copytree(src, destination)
        spec_path = destination / "Data/com.apple.CoreML/model.mlmodel"
        spec_path.write_bytes(variant.SerializeToString())
        assert wire(ct.models.utils.load_spec(str(destination))) == wire(variant)
        source_files, variant_files = inventory(src), inventory(destination)
        changed_files = [name for name in source_files if source_files[name] != variant_files.get(name)]
        assert source_files.keys() == variant_files.keys()
        assert changed_files == ["Data/com.apple.CoreML/model.mlmodel"]
        report.update(package=str(destination.relative_to(ROOT)), source_files=source_files,
                      variant_files=variant_files, changed_package_files=changed_files)
        save()
        if compile_package:
            from coreml_port.prediction_guard import install
            install()
            compiled = artifact / f"{group}.mlmodelc"
            started = time.perf_counter()
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", required=True)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    result = build(args.group, args.attempt, args.compile)
    print(json.dumps({"passed": result["passed"], "group": result["group"],
                      "moved_short_ops": result["graph_check"]["moved_short_ops"],
                      "compiled": result.get("compiled"),
                      "compile_seconds": result.get("compile_seconds")}, indent=2))


if __name__ == "__main__":
    main()
