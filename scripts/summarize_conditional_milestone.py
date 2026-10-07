"""Qualify a sealed single-model conditional score candidate; run no inference.

This adds a new arithmetic-policy gate. It does not relabel the previous
paired-native-tiny build or alter any frozen validator or acceptance limit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import summarize_cpu_milestone as base
from scripts.summarize_compiled_milestone import check_loader_provenance

POLICY = "conditional-warm-score"
ADAPTER = "compiled_tiny_prefix_replay"
# Pins are identities of the already-passing evidence, not today's source.
QUALIFIED = {
    "evidence/05c-compiled/milestone-rerun1.json": "8163c4b75acb946aae99d3a4a33880e6e2f826146afcbd556a8fd277b8ca6362",
    "evidence/05c-compiled/cohort-source-verification.json": "ca3d269f07b112ac60c0f2dc2877884588c124489bb4f5c212f3e987a67b45fe",
    "evidence/05c-compiled/short/validation-CPU_ONLY-attempt.json": "bc47479321b8eb54874b652631c164949ae38d3c74ef561074ae500e3f9d4810",
    "evidence/04d-audit/compiled1-generation/results.json": "1f4a8c3e0b456302619b9230e7a269454117f9e86b5831dd368da228ad30f274",
    "evidence/03-portable/calibration-cpu.json": "6c33c8976abdc5e2ef603519cdaa3f5193c443515de876dbea920b9840ab9d95",
    "evidence/04-coreml/generation-comparison.json": "1ee6553885123a57b8fb40b89c292eea201e148e6f3bd4d9b4cba1c057f9f3eb",
    "evidence/03b-cache/cpu-partitions-v3.json": "182ccd8c86d108b35465aa5cea44140a547cd45f0b502c20da20f426910be4a3",
    "evidence/04d-cpu/full-candidate-6/source-identity.json": "6e2b3ffd5c6a4e73f753e17ad1941181502df789dab4a1288456ebc7866137f4",
}


def qualified_context():
    """Read pinned prior evidence and measure only small source/guard files."""
    errors = []
    docs = {}
    for relative, digest in QUALIFIED.items():
        path = ROOT / relative
        if base.sha256(path) != digest:
            raise ValueError(f"qualified evidence identity differs: {relative}")
        docs[relative] = base.read_json(path)
    milestone, cohort, attempt, generation, calibration, comparison, _, baseline_sources = docs.values()
    if milestone.get("milestone_pass") is not True or cohort.get("passed") is not True:
        errors.append("pinned baseline milestone/cohort is not passing")
    expected = {}
    for mapping in (milestone["current_runtime_provenance"], cohort["expected_sha256"],
                    attempt["source_hashes"], attempt["loader_provenance"]["sources"],
                    generation["source_sha256"], baseline_sources):
        for path, digest in mapping.items():
            if path == "identity_manifest":
                continue
            if path in expected and expected[path] != digest:
                errors.append(f"qualified source records disagree: {path}")
            expected[path] = digest
    for relative, digest in expected.items():
        if base.sha256(ROOT / relative) != digest:
            errors.append(f"qualified runtime/validator/reference changed: {relative}")
    return {"milestone":milestone, "generation":generation,
            "limits":calibration["frozen_limits"], "reference":comparison["reference"],
            "expected_sources":expected, "errors":errors}


def require_generation_identity(result, context):
    """Require actual reference-identical tokens, sampling, EOS and disposal."""
    errors = []
    expected = context["reference"]["generated_ids"]
    generated = result.get("generated_ids")
    generation = result.get("generation", {})
    prior = context["generation"]
    if len(expected) != 674 or generated != expected:
        errors.append("generation is not exactly the 674 qualified reference tokens")
    if generation.get("generated_tokens") != 674:
        errors.append("generation count is not 674")
    if generation.get("common_prefix_with_qualified_cpu_tokens") != 674 or generation.get("qualified_cpu_token_count") != 674:
        errors.append("generation reference parity counters differ")
    if not isinstance(generated, list) or not generated or generated[-1] != 151643:
        errors.append("generation did not terminate with qualified explicit EOS 151643")
    if generation.get("finish_reason") != "stop_token":
        errors.append("generation finish reason differs from qualified stop_token")
    if generation.get("stop_token_ids") != [151643,151645] or generation.get("completion_policy") != "published_eos_plus_explicit_tokenizer_eos":
        errors.append("generation EOS/stop policy changed")
    if result.get("request_disposed_after_run") is not True or generation.get("request_disposed") is not True:
        errors.append("generation disposal evidence is incomplete")
    if result.get("fixture") != prior["fixture"] or result.get("qualified_reference_settings") != prior["qualified_reference_settings"]:
        errors.append("generation fixture/settings differ from qualified run")
    digest = hashlib.sha256(json.dumps(expected,separators=(",",":")).encode()).hexdigest()
    if result.get("qualified_reference_generated_ids_sha256") != digest:
        errors.append("generation reference token digest differs")
    for key in ("greedy","seed","temperature","top_p"):
        if generation.get(key) != prior["generation"][key]:
            errors.append(f"generation sampling setting changed: {key}")
    if len(result.get("audits", [])) != 7:
        errors.append("generation must contain exactly seven score audits")
    if result.get("limits") != context["limits"]:
        errors.append("generation limits differ from frozen CPU calibration")
    return errors


def candidate_result_binding(name, result, source_manifest, manifest_path):
    errors = []
    # Validators retain the complete strictly verified source manifest.
    observed = result.get("package_identity")
    if not isinstance(observed, dict):
        return ["full source package identity is missing"]
    if observed != source_manifest:
        errors.append(f"full source package manifest differs in {name}")
    expected_manifest_hash = base.sha256(manifest_path)
    if name == "generation":
        if result.get("source_sha256",{}).get("identity_manifest") != expected_manifest_hash:
            errors.append("generation source manifest file hash differs")
    return errors


def evaluate_gate(name, path, *, common, expected_loader, source_manifest,
                  manifest_path, context, attempt_path=None):
    if not path.is_file() or (attempt_path is not None and not attempt_path.is_file()):
        return {"status":"missing", "passed":False, "path":str(path),
                "errors":["selected gate evidence does not exist"]}
    try:
        result = base.read_json(path)
        if name == "short":
            attempt = base.read_json(attempt_path)
            gate = base.evaluate_short(result, attempt)
            adjusted = {**result, "script_sha256":attempt.get("script_sha256"),
                        "source_hashes":attempt.get("source_hashes",{})}
            errors = [*gate["errors"], *base.check_common(name, adjusted, **common)]
            errors.extend(check_loader_provenance(attempt, expected_loader, common["package_digest"]))
            if attempt.get("runtime_provenance") != common["current_runtime_provenance"] or attempt.get("runtime_adapter") != ADAPTER:
                errors.append("short attempt runtime identity/adapter differs")
            if base.result_model_dir(attempt) != str(common["candidate_root"]):
                errors.append("short attempt points to a different source candidate")
            if base.resolve_path(attempt.get("identity_manifest", "")) != manifest_path:
                errors.append("short attempt source manifest path differs")
            gate["attempt_path"] = str(attempt_path)
            gate["attempt_sha256"] = base.sha256(attempt_path)
        else:
            evaluators = {"partitions":base.evaluate_partitions,
                          "boundaries":base.evaluate_boundaries,
                          "generation":base.evaluate_generation}
            gate = base.load_gate(name,str(path),evaluators[name],**common)
            errors = gate["errors"]
        errors.extend(check_loader_provenance(result, expected_loader, common["package_digest"]))
        errors.extend(candidate_result_binding(name,result,source_manifest,manifest_path))
        if name in {"generation","partitions"} and result.get("limits") != context["limits"]:
            errors.append("recorded limits differ from frozen CPU calibration")
        if name == "partitions" and result.get("calibration_sha256") != QUALIFIED["evidence/03-portable/calibration-cpu.json"]:
            errors.append("partition calibration identity differs")
        if name == "generation":
            errors.extend(require_generation_identity(result,context))
        gate.update(status="pass" if not errors and gate["passed"] else "fail",
                    passed=not errors and gate["passed"] is True, errors=errors,
                    path=str(path), result_sha256=base.sha256(path))
        return gate
    except Exception as exc:
        return {"status":"invalid", "passed":False, "path":str(path),
                "errors":[f"{type(exc).__name__}: {exc}"]}


# Conditional build/seal schema has its own explicit policy. No fallback to
# the old paired-native-tiny build flags is accepted here.
BASELINE_MANIFEST = "evidence/04d-cpu/full-candidate-6/package-identity.json"
BASELINE_MANIFEST_SHA256 = "eaf5695cdbe1fe79185ad10942cc67567e3e760e0a8dc23e716b2bfd554b3d37"
BASELINE_IDENTITY = "5a3e4c7a571e4fc3532377d9b9bb0da023f499c10709216fee7e1001bf4ad2c6"
BUILDER = "scripts/build_conditional_scores.py"
ASSEMBLER = "scripts/assemble_conditional_model.py"


def inventory_by_package(manifest):
    return {entry["name"]:{item["path"].split("/",1)[1]:item["sha256"]
                           for item in entry["files"]}
            for entry in manifest["packages"]}


def check_conditional_build(evidence_dir, build, source_identity, source_manifest,
                            compiled_manifest):
    errors = []
    records = []
    baseline_path = ROOT / BASELINE_MANIFEST
    if base.sha256(baseline_path) != BASELINE_MANIFEST_SHA256:
        return [], ["baseline source package manifest differs from qualified identity"]
    baseline = base.read_json(baseline_path)
    baseline_files = inventory_by_package(baseline)
    variant_files = inventory_by_package(source_manifest)
    if build.get("status") != "completed" or build.get("passed") is not True or build.get("arithmetic_policy") != POLICY:
        errors.append("new conditional build status/arithmetic policy is missing or differs")
    if build.get("identity") != source_manifest.get("aggregate_sha256"):
        errors.append("conditional build identity differs from selected source manifest")
    if build.get("source_candidate_identity") != BASELINE_IDENTITY:
        errors.append("conditional build is not derived from qualified baseline6")
    if source_identity.get(BUILDER) != base.sha256(ROOT / BUILDER):
        errors.append("new conditional builder identity is missing or changed after sealing")
    if source_identity.get(ASSEMBLER) != base.sha256(ROOT / ASSEMBLER) or build.get("assembler_sha256") != source_identity.get(ASSEMBLER):
        errors.append("conditional assembler identity is missing or changed after sealing")
    if build.get("compiled_identity") != compiled_manifest.get("aggregate_sha256"):
        errors.append("assembled compiled identity differs from selected compiled manifest")
    copied = build.get("copied", {})
    if set(copied) != {"embedding","head"}:
        errors.append("conditional build must copy exactly embedding and head")
    for name in ("embedding","head"):
        if variant_files.get(name) != baseline_files[name]:
            errors.append(f"copied {name} package differs from qualified baseline")
    groups = build.get("groups", [])
    if not isinstance(groups,list):
        return records, [*errors,"conditional per-group records are missing"]
    if len(groups) != 12 or {x.get("group") for x in groups if isinstance(x,dict)} != base.GROUP_COMPONENTS:
        errors.append("conditional build does not bind exactly groups 0..11")
    for entry in groups:
        try:
            name = entry["group"]
            path = base.resolve_path(entry["record"])
            if base.sha256(path) != entry["record_sha256"]:
                raise ValueError("structural build record file hash differs")
            record = base.read_json(path)
            if record.get("status") != "completed" or record.get("passed") is not True or record.get("group") != name:
                raise ValueError("group build is incomplete, failed, or mislabelled")
            if record.get("source_candidate_identity") != BASELINE_IDENTITY:
                raise ValueError("group source candidate differs from qualified baseline")
            if record.get("script_sha256") != source_identity.get(BUILDER):
                raise ValueError("group builder digest differs from sealed new builder")
            snapshot = path.parent / "source-snapshot.py"
            if base.sha256(snapshot) != record.get("script_sha256"):
                raise ValueError("group builder snapshot digest differs")
            if record.get("source_files") != baseline_files[name] or record.get("variant_files") != variant_files[name]:
                raise ValueError("group source/variant inventories differ from selected manifests")
            changed_files = sorted(k for k,v in record["source_files"].items() if record["variant_files"].get(k) != v)
            if set(record["source_files"]) != set(record["variant_files"]) or changed_files != ["Data/com.apple.CoreML/model.mlmodel"]:
                raise ValueError("actual group file changes differ from exactly the model MIL spec")
            if record.get("changed_package_files") != ["Data/com.apple.CoreML/model.mlmodel"]:
                raise ValueError("group changed package files exceed model MIL spec")
            # Build-time compiled inventories must survive any final assembly copy.
            if "compiled_files" not in record or not record.get("compiled"):
                raise ValueError("group compiled inventory is missing")
            if "compiled_files" in record:
                compiled_entry = compiled_manifest["packages"][name]
                expected_files = {x["path"].split("/",1)[1]:x["sha256"] for x in compiled_entry["files"]}
                if record["compiled_files"] != expected_files:
                    raise ValueError("group compiled inventory differs from selected compiled manifest")
            graph = record.get("graph_check", {})
            flags = ("passed","all_moved_ops_byte_identical","all_non_operation_metadata_identical",
                     "value_product_matmuls_unchanged","state_reads_unchanged","state_writes_unchanged",
                     "single_model_state_interface_preserved")
            if any(graph.get(flag) is not True for flag in flags):
                raise ValueError("group graph/state/arithmetic invariants are missing or failed")
            expected_counts = {"selector_rewrites":4,"exclusive_short_ops":10009,
                               "moved_short_ops":10004,"shared_short_ops_kept_outside":5,
                               "true_block_ops":[2502]*4,"new_branch_shape_refinement_identities":4,"new_false_branch_return_identities":4}
            if any(graph.get(key) != value for key,value in expected_counts.items()):
                raise ValueError("conditional rewrite operation counts differ")
            before, after = graph["before"], graph["after_all_blocks"]
            for operation in ("matmul","read_state","write_state"):
                if not isinstance(before.get(operation),int) or before[operation] <= 0 or after.get(operation) != before[operation]:
                    raise ValueError(f"group {operation} counts changed")
            expected_after = dict(before)
            expected_after["select"] -= 4
            expected_after["cond"] = expected_after.get("cond",0) + 4
            expected_after["identity"] = expected_after.get("identity",0) + 8
            if after != expected_after:
                raise ValueError("operations changed beyond four select-to-cond rewrites and eight branch identities")
            if graph.get("unchanged_surviving_ops") != sum(before.values()) - 10004 - 4:
                raise ValueError("surviving operation count is inconsistent")
            selectors = graph.get("selectors", [])
            if len(selectors) != 4 or len({x.get("output") for x in selectors}) != 4:
                raise ValueError("group selector set is missing or duplicated")
            for selector in selectors:
                if selector.get("predicate_rule") != "prepared_query_rows < 16" or selector.get("threshold") != 16:
                    raise ValueError("group score predicate differs")
                if selector.get("true_block_ops") != 2502 or selector.get("false_block_ops") != 1:
                    raise ValueError("group score branch operation counts differ")
                if selector.get("ordinary_type_sha256") != selector.get("output_type_sha256") or not selector.get("shape_refinement", "").endswith("_conditional_refined") or not selector.get("false_block_output") or selector.get("false_block_output") == selector.get("ordinary_scores"):
                    raise ValueError("group branch shape refinement is missing or inconsistent")
                for key in ("short_type_sha256","ordinary_type_sha256","output_type_sha256","true_block_sha256"):
                    if not re.fullmatch(r"[0-9a-f]{64}",selector.get(key,"")):
                        raise ValueError("group branch/type hash missing")
                if not selector.get("short_scores","").startswith("short_scores") or not selector.get("ordinary_scores","").startswith("full_scores") or not selector.get("predicate"):
                    raise ValueError("group score branch/predicate names missing")
            records.append({"group":name,"record":str(path),"record_sha256":entry["record_sha256"],
                            "selectors":4,"moved_short_ops":10004,"passed":True})
        except Exception as exc:
            errors.append(f"conditional group {entry.get('group','?') if isinstance(entry,dict) else '?'}: {type(exc).__name__}: {exc}")
    return records, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root",required=True)
    parser.add_argument("--candidate-evidence-dir",required=True)
    parser.add_argument("--compiled-dir",required=True)
    parser.add_argument("--compiled-manifest",required=True)
    for name in ("short-result","short-attempt","partitions-result","boundaries-result","generation-result","output"):
        parser.add_argument("--"+name,required=True)
    args = parser.parse_args()
    path = lambda key:base.resolve_path(getattr(args,key))
    output = path("output")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite milestone summary: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"Output parent must exist: {output.parent}")
    candidate_root,evidence_dir = path("candidate_root"),path("candidate_evidence_dir")
    manifest_path = evidence_dir / "package-identity.json"
    errors = []
    context = {}
    try:
        context = qualified_context()
        errors.extend(context["errors"])
    except Exception as exc:
        errors.append(f"qualified context: {type(exc).__name__}: {exc}")
    source_manifest = {}; package_digest = None; seal = {}; structural = []
    compiled = {}; expected_loader = {}; compiled_errors = []
    try:
        source_manifest = base.read_json(manifest_path)
        package_digest = source_manifest["aggregate_sha256"]
        if [x["name"] for x in source_manifest["packages"]] != list(base.EXPECTED_PACKAGES):
            errors.append("source manifest does not contain exactly the ordered 14 packages")
        if not re.fullmatch(r"[0-9a-f]{64}",package_digest):
            errors.append("source package aggregate identity is malformed")
        build = base.read_json(evidence_dir / "build.json")
        source_identity = base.read_json(evidence_dir / "source-identity.json")
        if not source_identity or any(not re.fullmatch(r"[0-9a-f]{64}",v) for v in source_identity.values()):
            errors.append("conditional source identities are empty or malformed")
        seal, seal_errors = base.verify_sealed_provenance(evidence_dir,build,source_manifest,source_identity)
        errors.extend(seal_errors)
        compiled_manifest = base.read_json(path("compiled_manifest"))
        structural, structural_errors = check_conditional_build(evidence_dir,build,source_identity,source_manifest,compiled_manifest)
        errors.extend(structural_errors)
        if base.resolve_path(build.get("source_root", "")) != candidate_root or base.resolve_path(build.get("compiled_root", "")) != path("compiled_dir"):
            errors.append("assembly roots differ from explicitly selected source/cache")
    except Exception as exc:
        errors.append(f"conditional source provenance: {type(exc).__name__}: {exc}")
    try:
        # This is the only heavyweight step: actual strict source/cache bytes
        # plus environment are reverified, using the unchanged qualified helper.
        from coreml_port.compiled_identity import verify_compiled_identity
        from coreml_port.loader_selection import loader_provenance
        compiled = verify_compiled_identity(path("compiled_dir"),path("compiled_manifest"),candidate_root,manifest_path)
        loader_args = argparse.Namespace(compiled_dir=str(path("compiled_dir")),
            compiled_manifest=str(path("compiled_manifest")),tiny_prefix_replay=True,
            compute="CPU_ONLY",model_dir=str(candidate_root),identity_manifest=str(manifest_path))
        expected_loader = loader_provenance(loader_args)
        if compiled.get("source_candidate_identity") != package_digest or expected_loader.get("source_candidate_identity") != package_digest:
            compiled_errors.append("compiled cache is bound to a different source candidate")
        if expected_loader.get("compiled_aggregate_sha256") != compiled.get("aggregate_sha256"):
            compiled_errors.append("loader and verified compiled aggregate differ")
    except Exception as exc:
        compiled_errors.append(f"compiled identity: {type(exc).__name__}: {exc}")
    current_runtime = base.runtime_provenance()
    common = dict(candidate_root=candidate_root,package_digest=package_digest,adapter=ADAPTER,
        current_adapter_sha256=base.sha256(ROOT / "coreml_port/tiny_prefix_runtime.py"),
        current_runtime_provenance=current_runtime,acceptance_sha256=base.FROZEN_ACCEPTANCE_SHA256)
    gates = {}
    for name in ("short","partitions","boundaries","generation"):
        if not context:
            gates[name] = {"status":"invalid","passed":False,"errors":["qualified reference context unavailable"]}
            continue
        gates[name] = evaluate_gate(name,path(name+"_result"),common=common,
            expected_loader=expected_loader,source_manifest=source_manifest,manifest_path=manifest_path,
            context=context,attempt_path=path("short_attempt") if name=="short" else None)
    missing = [name for name,gate in gates.items() if gate["status"] == "missing"]
    failed = (["conditional_source_provenance"] if errors else []) + (["compiled_identity"] if compiled_errors else []) + [name for name,gate in gates.items() if not gate["passed"]]
    passed = not failed
    result = {"schema_version":1,"status":"completed" if passed else "incomplete" if missing else "completed_with_failures",
        "milestone_pass":passed,"thresholds_changed":False,"compute":"CPU_ONLY",
        "selected_arithmetic_policy":POLICY,"selected_runtime_adapter":ADAPTER,
        "acceptance_sha256":base.FROZEN_ACCEPTANCE_SHA256,"qualified_evidence_sha256":QUALIFIED,
        "qualified_source_sha256":context.get("expected_sources",{}),
        "current_runtime_provenance":current_runtime,"aggregator_sha256":base.sha256(Path(__file__)),
        "aggregator_dependencies_sha256":{p:base.sha256(ROOT/p) for p in
            ("scripts/summarize_cpu_milestone.py","scripts/summarize_compiled_milestone.py","scripts/runtime_provenance.py")},
        "source_provenance":{"passed":not errors,"status":"pass" if not errors else "fail",
            "candidate_root":str(candidate_root),"candidate_evidence_dir":str(evidence_dir),
            "source_package_aggregate_sha256":package_digest,"manifest_sha256":base.sha256(manifest_path) if manifest_path.is_file() else None,
            "seal":seal,"groups":structural,"errors":errors},
        "compiled_identity":{"passed":not compiled_errors,"status":"pass" if not compiled_errors else "fail",
            "compiled_dir":str(path("compiled_dir")),"compiled_manifest":str(path("compiled_manifest")),
            "verified_identity":compiled,"loader_provenance":expected_loader,"errors":compiled_errors},
        "gates":gates,"missing_gates":missing,"failed_gates":failed}
    output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":result["status"],"milestone_pass":passed,"failed_gates":failed,"output":str(output)},indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
