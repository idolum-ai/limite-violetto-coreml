"""Aggregate candidate-bound CPU Core ML gate evidence without running inference."""

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
from coreml_port.package_identity import EXPECTED_PACKAGES
from scripts.runtime_provenance import runtime_provenance

FROZEN_ACCEPTANCE_SHA256 = "cb4d315266126df4a98194f73f8c10671310717c06ebfd191d3dac2334bf4233"
EXPECTED_BOUNDARY_LABELS = {
    "prefix1023", "prefix1024", "prefix1025", "prefix1026", "prefix2048", "fresh1026"
}
EXPECTED_PARTITION_CASES = {
    "single_prefill_1": 1,
    "single_prefill_3": 1,
    "single_prefill_15": 1,
    "single_prefill_16": 1,
    "single_prefill_31": 1,
    "single_prefill_32": 1,
    "single_prefill_33": 1,
    "single_token_partitions_1": 1,
    "single_token_partitions_3": 3,
    "single_token_partitions_15": 15,
    "single_token_partitions_31": 31,
    "single_token_partitions_33": 33,
    "irregular_127": 7,
}
ARITHMETIC_POLICIES = {"legacy-dynamic-product", "paired-product", "paired-tiny", "paired-native-tiny"}
GROUP_COMPONENTS = {str(index) for index in range(12)}
PAIRED_PRODUCT_HELPER = "coreml_port/paired_product.py"
TINY_SCORE_HELPER = "coreml_port/tiny_scores.py"
PAIRED_TINY_HELPERS = {
    PAIRED_PRODUCT_HELPER,
    TINY_SCORE_HELPER,
}
TINY_NATIVE_SCORE_HELPER = "coreml_port/tiny_native_scores.py"
PAIRED_NATIVE_TINY_HELPERS = {
    PAIRED_PRODUCT_HELPER,
    TINY_NATIVE_SCORE_HELPER,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def identity_digest(result: dict[str, Any]) -> str | None:
    value = result.get("package_identity")
    return value.get("aggregate_sha256") if isinstance(value, dict) else None


def result_model_dir(result: dict[str, Any]) -> str | None:
    value = result.get("model_dir") or result.get("candidate_dir")
    return str(resolve_path(value)) if isinstance(value, str) else None


def adapter_hash_for(gate_name: str, value: dict[str, Any]) -> str | None:
    if gate_name == "short":
        return value.get("source_hashes", {}).get("coreml_port/tiny_prefix_runtime.py")
    if gate_name == "boundaries":
        return value.get("source_sha256", {}).get("coreml_port/tiny_prefix_runtime.py")
    if gate_name == "partitions":
        return value.get("tiny_prefix_runtime_sha256")
    if gate_name == "generation":
        return value.get("source_sha256", {}).get("coreml_port/tiny_prefix_runtime.py")
    return None


def check_common(
    name: str,
    result: dict[str, Any],
    *,
    candidate_root: Path,
    package_digest: str,
    adapter: str,
    current_adapter_sha256: str,
    current_runtime_provenance: dict[str, str],
    acceptance_sha256: str,
) -> list[str]:
    errors: list[str] = []
    actual_digest = identity_digest(result)
    if actual_digest != package_digest:
        errors.append("package identity differs from candidate manifest")
    actual_root = result_model_dir(result)
    if actual_root != str(candidate_root):
        errors.append(f"model root differs from selected candidate ({actual_root!r})")
    if result.get("runtime_adapter") != adapter:
        errors.append(f"runtime adapter is missing or differs from expected {adapter!r}")
    if adapter_hash_for(name, result) != current_adapter_sha256:
        errors.append("tiny-prefix runtime source hash is missing or stale")
    if result.get("runtime_provenance") != current_runtime_provenance:
        errors.append("runtime/cleanup/prediction guard identities are missing or stale")
    accepted = result.get("acceptance_sha256")
    if name == "short":
        accepted = result.get("source_hashes", {}).get("evidence/03-portable/ACCEPTANCE.json")
    if name == "partitions" and accepted is None:
        accepted = result.get("acceptance_sha256")
    if accepted != acceptance_sha256:
        errors.append("frozen acceptance digest differs or is missing")
    if result.get("thresholds_changed") is not False:
        errors.append("threshold identity is missing or thresholds_changed is not false")
    validator_paths = {
        "short": "scripts/validate_coreml.py",
        "boundaries": "scripts/validate_coreml_boundaries.py",
        "partitions": "scripts/audit_coreml_partitions.py",
        "generation": "scripts/audit_coreml_generation.py",
    }
    validator_key = "validator_sha256" if name == "partitions" else "script_sha256"
    recorded_validator = result.get(validator_key)
    if name == "generation":
        recorded_validator = result.get("source_sha256", {}).get(validator_paths[name])
    if name == "short":
        recorded_validator = result.get("script_sha256")
    if recorded_validator != sha256(ROOT / validator_paths[name]):
        errors.append("validator source hash is missing or stale")
    if result.get("compute", result.get("compute_units")) != "CPU_ONLY":
        errors.append("gate was not recorded as CPU_ONLY")
    return errors


def evaluate_short(result: dict[str, Any], attempt: dict[str, Any]) -> dict[str, Any]:
    comparisons = result.get("comparisons")
    checks = result.get("checks")
    comparison_count = len(comparisons) if isinstance(comparisons, list) else 0
    check_count = len(checks) if isinstance(checks, dict) else 0
    numerical_ok = (
        comparison_count == 54
        and all(isinstance(item, dict) and item.get("pass") is True for item in comparisons)
        and result.get("numerical_pass") is True
    )
    lifecycle_ok = (
        check_count == 17
        and all(value is True for value in checks.values())
        and result.get("lifecycle_pass") is True
    )
    score_rows = [item for item in comparisons if isinstance(item, dict) and ("/scores" in str(item.get("name", "")) or "/teacher" in str(item.get("name", "")))] if isinstance(comparisons, list) else []
    hidden_rows = [item for item in comparisons if isinstance(item, dict) and "/scores" not in str(item.get("name", "")) and "/teacher" not in str(item.get("name", ""))] if isinstance(comparisons, list) else []
    return {
        "status": "pass" if numerical_ok and lifecycle_ok and attempt.get("status") == "completed" else "fail",
        "passed": numerical_ok and lifecycle_ok and attempt.get("status") == "completed",
        "attempt_status": attempt.get("status"),
        "comparison_count": comparison_count,
        "expected_comparisons": 54,
        "numerical_pass": numerical_ok,
        "check_count": check_count,
        "expected_checks": 17,
        "lifecycle_pass": lifecycle_ok,
        "max_score_abs": max((item.get("max_abs", 0.0) for item in score_rows), default=None),
        "max_hidden_relative_rmse": max((item.get("relative_rmse", 0.0) for item in hidden_rows), default=None),
        "errors": [] if numerical_ok and lifecycle_ok and attempt.get("status") == "completed" else ["short numerical/lifecycle count or status mismatch"],
    }


def evaluate_boundaries(result: dict[str, Any]) -> dict[str, Any]:
    comparisons = result.get("comparisons")
    checks = result.get("checks")
    labels = {item.get("name") for item in comparisons if isinstance(item, dict)} if isinstance(comparisons, list) else set()
    errors = []
    if result.get("status") != "completed":
        errors.append("boundary run is not completed")
    if labels != EXPECTED_BOUNDARY_LABELS or len(comparisons or []) != len(EXPECTED_BOUNDARY_LABELS):
        errors.append("boundary comparison set is incomplete or duplicated")
    if not isinstance(comparisons, list) or not all(item.get("pass") is True for item in comparisons if isinstance(item, dict)):
        errors.append("one or more boundary comparisons failed")
    if not isinstance(checks, dict) or len(checks) != 4 or not all(value is True for value in checks.values()):
        errors.append("boundary lifecycle/capacity checks are incomplete or failed")
    if result.get("numerical_pass") is not True or result.get("checks_pass") is not True:
        errors.append("boundary summary flags do not pass")
    return {"status": "pass" if not errors else "fail", "passed": not errors,
            "comparison_count": len(comparisons) if isinstance(comparisons, list) else 0,
            "labels": sorted(labels), "check_count": len(checks) if isinstance(checks, dict) else 0,
            "max_score_abs": max((item.get("scores", {}).get("max_abs", 0.0) for item in (comparisons or []) if isinstance(item, dict)), default=None),
            "max_hidden_relative_rmse": max((item.get("hidden", {}).get("relative_rmse", 0.0) for item in (comparisons or []) if isinstance(item, dict)), default=None),
            "errors": errors}


def evaluate_partitions(result: dict[str, Any]) -> dict[str, Any]:
    cases = result.get("cases")
    errors = []
    by_name = {case.get("name"): case for case in cases if isinstance(case, dict)} if isinstance(cases, list) else {}
    if result.get("status") != "completed":
        errors.append("partition run is not completed")
    if set(by_name) != set(EXPECTED_PARTITION_CASES) or len(cases or []) != len(EXPECTED_PARTITION_CASES):
        errors.append("partition case set is incomplete or duplicated")
    step_count = 0
    passed_steps = 0
    disposed_count = 0
    for name, expected_steps in EXPECTED_PARTITION_CASES.items():
        case = by_name.get(name)
        if case is None:
            continue
        steps = case.get("matched_prefixes")
        steps = steps if isinstance(steps, list) else []
        step_count += len(steps)
        passed_steps += sum(isinstance(step, dict) and step.get("passed") is True for step in steps)
        if len(steps) != expected_steps:
            errors.append(f"{name}: expected {expected_steps} matched prefixes, got {len(steps)}")
        if not case.get("passed") or not all(isinstance(step, dict) and step.get("passed") is True for step in steps):
            errors.append(f"{name}: case or matched-prefix checks failed")
        if case.get("request_disposed") is True:
            disposed_count += 1
        else:
            errors.append(f"{name}: request disposal missing or failed")
    repeat = result.get("partition_repeat")
    repeat_ok = isinstance(repeat, dict) and all(repeat.get(key) is True for key in ["scores_exact", "hidden_exact", "requests_disposed"])
    if not repeat_ok or result.get("partition_repeat_pass") is not True:
        errors.append("partition repeat/disposal gate failed")
    if result.get("numerical_pass") is not True or result.get("passed") is not True:
        errors.append("partition numerical summary does not pass")
    if step_count != 97:
        errors.append(f"expected 97 matched-prefix steps, got {step_count}")
    return {"status": "pass" if not errors else "fail", "passed": not errors,
            "case_count": len(by_name), "matched_prefix_count": step_count,
            "expected_matched_prefix_count": 97, "passed_prefix_count": passed_steps,
            "disposed_request_count": disposed_count,
            "max_score_abs": max((s.get("scores", {}).get("max_abs", 0.0) for c in (cases or []) if isinstance(c, dict) for s in c.get("matched_prefixes", []) if isinstance(s, dict)), default=None),
            "max_hidden_relative_rmse": max((s.get("hidden", {}).get("relative_rmse", 0.0) for c in (cases or []) if isinstance(c, dict) for s in c.get("matched_prefixes", []) if isinstance(s, dict)), default=None),
            "partition_repeat_pass": bool(repeat_ok), "errors": errors}


def evaluate_generation(result: dict[str, Any]) -> dict[str, Any]:
    audits = result.get("audits")
    generation = result.get("generation")
    errors = []
    if result.get("status") != "completed":
        errors.append("generation audit is not completed")
    if not isinstance(audits, list) or not audits or not all(
        isinstance(audit, dict) and audit.get("scores", {}).get("passed") is True for audit in (audits or [])
    ):
        errors.append("generation score audits are missing or failed")
    if result.get("audits_pass") is not True:
        errors.append("generation audit summary does not pass")
    if not isinstance(generation, dict):
        errors.append("generation result is missing")
        generation = {}
    for key in ["complete", "boxed_391_after_think", "think_marker_present", "qualified_reference_settings_match", "request_disposed"]:
        if generation.get(key) is not True:
            errors.append(f"generation {key} is missing or false")
    token_count = generation.get("generated_tokens")
    if isinstance(token_count, int) and token_count > 0 and isinstance(audits, list):
        expected_labels = {f"periodic_{step}" for step in range(0, token_count, 128)} | {"terminal"}
        observed_labels = [item.get("label") for item in audits if isinstance(item, dict)]
        if len(observed_labels) != len(expected_labels) or set(observed_labels) != expected_labels:
            errors.append("generation audit checkpoints are incomplete or duplicated")
    else:
        errors.append("generated token count is missing or invalid")
    if result.get("generation_pass") is not True or result.get("passed") is not True:
        errors.append("generation/task summary does not pass")
    return {"status": "pass" if not errors else "fail", "passed": not errors,
            "audit_count": len(audits) if isinstance(audits, list) else 0,
            "max_audited_score_error": result.get("max_audited_score_error"),
            "generated_tokens": generation.get("generated_tokens"),
            "finish_reason": generation.get("finish_reason"),
            "think_marker_present": generation.get("think_marker_present"),
            "boxed_391_after_think": generation.get("boxed_391_after_think"),
            "request_disposed": generation.get("request_disposed"), "errors": errors}


def load_gate(name: str, value: str | None, evaluate, **common: Any) -> dict[str, Any]:
    if value is None:
        return {"status": "missing", "passed": False, "errors": ["gate evidence path not supplied"]}
    path = resolve_path(value)
    try:
        result = read_json(path)
        summary = evaluate(result)
        errors = [*summary.get("errors", []), *check_common(name, result, **common)]
        if name == "partitions" and result.get("portable_partition_fixture_sha256") != sha256(ROOT / "evidence/03b-cache/cpu-partitions-v3.json"):
            errors.append("portable partition fixture digest differs or is missing")
        if name == "boundaries":
            expected_case = json.loads((ROOT / "evidence/01-reference/local-input-validation.json").read_text())['cases'][0]['input_ids']
            canonical_prefix = (expected_case * 24)[:1026]
            expected_ids = canonical_prefix + [100] * 1022
            expected_ids_hash = hashlib.sha256(json.dumps(expected_ids).encode()).hexdigest()
            if result.get("fixture") != "canonical-cycle" or result.get("input_ids_sha256") != expected_ids_hash:
                errors.append("boundary fixture is not the frozen canonical-cycle sequence")
        if name == "short":
            if result.get("comparisons") and resolve_path(result.get("model_dir", "")) != common["candidate_root"]:
                errors.append("short model root does not match candidate root")
        if name == "generation":
            source_hashes = result.get("source_sha256", {})
            fixture_hashes = {
                "evidence/04-coreml/generation-cpu.json": sha256(ROOT / "evidence/04-coreml/generation-cpu.json"),
                "evidence/04-coreml/generation-comparison.json": sha256(ROOT / "evidence/04-coreml/generation-comparison.json"),
            }
            if any(source_hashes.get(key) != digest for key, digest in fixture_hashes.items()):
                errors.append("generation fixture/reference source hashes differ or are missing")
        summary.update(path=str(path), status="pass" if not errors and summary.get("passed") else "fail",
                       passed=not errors and summary.get("passed") is True, errors=errors)
        return summary
    except Exception as exc:
        return {"status": "invalid", "passed": False, "path": str(path),
                "errors": [f"cannot evaluate gate evidence: {type(exc).__name__}: {exc}"]}


def verify_sealed_provenance(
    evidence_dir: Path,
    build: dict[str, Any],
    manifest: dict[str, Any],
    source_identity: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Require sealed source, snapshot, and copied-package lineage evidence."""
    errors: list[str] = []
    verification_path = evidence_dir / "provenance-verification.json"
    try:
        seal = read_json(verification_path)
    except Exception as exc:
        return {}, [f"cannot load sealed provenance verification: {type(exc).__name__}: {exc}"]

    digest = manifest.get("aggregate_sha256")
    before = seal.get("before")
    after = seal.get("after")
    if seal.get("pass") is not True:
        errors.append("sealed provenance verification pass flag is false or missing")
    if seal.get("sources_unchanged_during_build") is not True:
        errors.append("sealed provenance says sources changed during build")
    if before != source_identity:
        errors.append("sealed before-source identity differs from source-identity.json")
    if after != before:
        errors.append("sealed after-source identity differs from sealed before identity")
    if seal.get("package_identity") != digest:
        errors.append("sealed package identity differs from candidate manifest")

    expected_copies = build.get("copied", {})
    copied_lineage = seal.get("copied_lineage")
    if not isinstance(expected_copies, dict) or not isinstance(copied_lineage, list):
        errors.append("copied package lineage is malformed")
        expected_copies = expected_copies if isinstance(expected_copies, dict) else {}
        copied_lineage = copied_lineage if isinstance(copied_lineage, list) else []
    lineage = {entry.get("component"): entry for entry in copied_lineage if isinstance(entry, dict)}
    if set(lineage) != set(expected_copies) or len(copied_lineage) != len(expected_copies):
        errors.append("sealed copied-lineage components differ from build copied packages")
    for component, source_package in expected_copies.items():
        entry = lineage.get(component)
        if entry is None:
            continue
        if entry.get("source_package") != source_package:
            errors.append(f"{component}: copied-lineage source differs from build record")
        if entry.get("pass") is not True or entry.get("same_file_set") is not True:
            errors.append(f"{component}: copied lineage did not pass")
        files = entry.get("files")
        if not isinstance(files, list) or not files or not all(
            isinstance(item, dict)
            and item.get("size_matches") is True
            and isinstance(item.get("sha256"), str)
            and item.get("sha256") == item.get("expected")
            for item in files
        ):
            errors.append(f"{component}: copied files do not match their recorded source hashes")

    snapshot_root = (evidence_dir / "source-snapshots").resolve()
    snapshot_count = 0
    for relative, expected_hash in source_identity.items():
        relpath = Path(relative)
        if relpath.is_absolute() or ".." in relpath.parts or not relpath.parts:
            errors.append(f"unsafe source snapshot path: {relative!r}")
            continue
        snapshot = (snapshot_root / relpath).resolve()
        if not snapshot.is_relative_to(snapshot_root) or not snapshot.is_file():
            errors.append(f"source snapshot missing: {relative}")
        elif sha256(snapshot) != expected_hash:
            errors.append(f"source snapshot digest differs: {relative}")
        else:
            snapshot_count += 1
    if snapshot_count != len(source_identity):
        errors.append("not all source snapshots match source-identity.json")

    return {
        "path": str(verification_path),
        "sources_unchanged_during_build": seal.get("sources_unchanged_during_build"),
        "before_matches_source_identity": before == source_identity,
        "after_matches_before": after == before,
        "package_identity_matches": seal.get("package_identity") == digest,
        "copied_components": sorted(lineage),
        "snapshot_count": snapshot_count,
    }, errors


def check_arithmetic_policy(
    policy: str,
    build: dict[str, Any],
    source_identity: dict[str, Any],
) -> list[str]:
    """Bind candidate arithmetic mode to sealed helpers and conversion commands."""
    errors: list[str] = []
    if policy not in ARITHMETIC_POLICIES:
        return [f"unsupported arithmetic policy: {policy!r}"]

    paired = build.get("paired_product") is True
    tiny = build.get("tiny_sequential") is True
    native_tiny = build.get("tiny_native") is True
    conversions = build.get("conversions")

    def commands_have(option: str, value: str) -> bool:
        if not isinstance(conversions, list):
            return False
        return any(
            isinstance(item, dict)
            and isinstance(item.get("command"), list)
            and any(item["command"][index:index + 2] == [option, value]
                    for index in range(len(item["command"]) - 1))
            for item in conversions
        )

    if policy == "legacy-dynamic-product":
        if paired or tiny or native_tiny:
            errors.append("legacy arithmetic policy selected for a paired-tiny build")
        if commands_have("--score-dot", "tiny-native"):
            errors.append("legacy arithmetic policy forbids --score-dot tiny-native")
        return errors

    if policy == "paired-product":
        if paired is not True:
            errors.append("paired-product policy requires build paired_product=true")
        if tiny:
            errors.append("paired-product policy forbids build tiny_sequential=true")
        if native_tiny:
            errors.append("paired-product policy forbids build tiny_native=true")
        digest = source_identity.get(PAIRED_PRODUCT_HELPER)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            errors.append(f"paired-product source identity is missing or malformed: {PAIRED_PRODUCT_HELPER}")
        conversions = build.get("conversions")
        if not isinstance(conversions, list):
            errors.append("paired-product conversion command records are missing")
            return errors
        seen_groups: set[str] = set()
        for item in conversions:
            if not isinstance(item, dict) or item.get("component") not in GROUP_COMPONENTS:
                continue
            component = item["component"]
            seen_groups.add(component)
            command = item.get("command")
            if not isinstance(command, list):
                errors.append(f"group {component}: paired-product conversion command is missing")
                continue
            if not any(command[index:index + 2] == ["--product", "paired"] for index in range(len(command) - 1)):
                errors.append(f"group {component}: conversion command lacks --product paired")
            if any(command[index:index + 2] == ["--score-dot", "tiny-sequential"] for index in range(len(command) - 1)):
                errors.append(f"group {component}: paired-product policy forbids --score-dot tiny-sequential")
            if any(command[index:index + 2] == ["--score-dot", "tiny-native"] for index in range(len(command) - 1)):
                errors.append(f"group {component}: paired-product policy forbids --score-dot tiny-native")
        missing_groups = GROUP_COMPONENTS - seen_groups
        if missing_groups:
            errors.append(f"paired-product conversion commands missing groups: {sorted(missing_groups)}")
        return errors

    native_policy = policy == "paired-native-tiny"
    if paired is not True:
        errors.append(f"{policy} policy requires build paired_product=true")
    if native_policy:
        if native_tiny is not True:
            errors.append("paired-native-tiny policy requires build tiny_native=true")
        if tiny:
            errors.append("paired-native-tiny policy forbids build tiny_sequential=true")
        for option, label in (("enhanced", "enhanced"), ("source_exp", "source_exp"),
                              ("dynamic_product", "dynamic_product")):
            if build.get(option) is not True:
                errors.append(f"paired-native-tiny policy requires build {label}=true")
        helpers = PAIRED_NATIVE_TINY_HELPERS
        score_dot = "tiny-native"
    else:
        if tiny is not True:
            errors.append("paired-tiny policy requires build tiny_sequential=true")
        if native_tiny:
            errors.append("paired-tiny policy forbids build tiny_native=true")
        helpers = PAIRED_TINY_HELPERS
        score_dot = "tiny-sequential"

    for helper in sorted(helpers):
        digest = source_identity.get(helper)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            errors.append(f"{policy} source identity is missing or malformed: {helper}")

    if not isinstance(conversions, list):
        errors.append(f"{policy} conversion command records are missing")
        return errors
    seen_groups: set[str] = set()
    for item in conversions:
        if not isinstance(item, dict) or item.get("component") not in GROUP_COMPONENTS:
            continue
        component = item["component"]
        seen_groups.add(component)
        command = item.get("command")
        if not isinstance(command, list):
            errors.append(f"group {component}: {policy} conversion command is missing")
            continue
        if not any(command[index:index + 2] == ["--product", "paired"] for index in range(len(command) - 1)):
            errors.append(f"group {component}: conversion command lacks --product paired")
        if not any(command[index:index + 2] == ["--score-dot", score_dot] for index in range(len(command) - 1)):
            errors.append(f"group {component}: conversion command lacks --score-dot {score_dot}")
        wrong_dot = "tiny-sequential" if native_policy else "tiny-native"
        if any(command[index:index + 2] == ["--score-dot", wrong_dot] for index in range(len(command) - 1)):
            errors.append(f"group {component}: {policy} policy forbids --score-dot {wrong_dot}")
    missing_groups = GROUP_COMPONENTS - seen_groups
    if missing_groups:
        errors.append(f"paired-tiny conversion commands missing groups: {sorted(missing_groups)}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", required=True, help="Explicit 14-package candidate root.")
    parser.add_argument("--candidate-evidence-dir", required=True, help="Build and identity evidence directory.")
    parser.add_argument("--short-result", required=True, help="Full short-suite validation JSON.")
    parser.add_argument("--short-attempt", required=True, help="Companion short-suite attempt/provenance JSON.")
    parser.add_argument("--runtime-adapter", required=True, choices=["standard_pipeline", "tiny_prefix_replay"])
    parser.add_argument("--arithmetic-policy", choices=sorted(ARITHMETIC_POLICIES),
                        default="legacy-dynamic-product",
        help="Require provenance for legacy, paired-product, paired tiny-sequential, or paired native tiny-score arithmetic.")
    parser.add_argument("--partitions-result", help="Optional tiny/irregular partition audit JSON.")
    parser.add_argument("--boundaries-result", help="Optional long boundary/capacity audit JSON.")
    parser.add_argument("--generation-result", help="Optional sampled generation audit JSON.")
    parser.add_argument("--output", required=True, help="New summary JSON path; existing files are not overwritten.")
    args = parser.parse_args()

    candidate_root = resolve_path(args.candidate_root)
    evidence_dir = resolve_path(args.candidate_evidence_dir)
    output_path = resolve_path(args.output)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite milestone summary: {output_path}")
    if not output_path.parent.is_dir():
        raise FileNotFoundError(f"Output parent directory must already exist: {output_path.parent}")

    manifest_path = evidence_dir / "package-identity.json"
    build_path = evidence_dir / "build.json"
    source_path = evidence_dir / "source-identity.json"
    provenance_errors: list[str] = []
    seal_summary: dict[str, Any] = {}
    try:
        manifest = read_json(manifest_path)
        build = read_json(build_path)
        source_identity = read_json(source_path)
        digest = manifest.get("aggregate_sha256")
        names = [item.get("name") for item in manifest.get("packages", []) if isinstance(item, dict)]
        if names != list(EXPECTED_PACKAGES):
            provenance_errors.append("package identity manifest does not list the ordered expected 14 packages")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            provenance_errors.append("package identity aggregate digest is malformed")
        if build.get("identity") != digest:
            provenance_errors.append("build identity differs from package manifest")
        if not isinstance(build.get("status"), str) or not build["status"].startswith("converted"):
            provenance_errors.append("candidate build is not marked converted")
        if build.get("enhanced") is not True or build.get("source_exp") is not True or build.get("dynamic_product") is not True:
            provenance_errors.append("candidate build lacks the qualified enhanced/source-exp/dynamic-product options")
        copied = build.get("copied", {})
        conversions = build.get("conversions", [])
        described = list(copied) + [item.get("component") for item in conversions if isinstance(item, dict)]
        if set(described) != set(EXPECTED_PACKAGES) or len(described) != len(EXPECTED_PACKAGES):
            provenance_errors.append("build does not describe each expected package exactly once")
        if not all(isinstance(item, dict) and item.get("exit_code") == 0 for item in conversions):
            provenance_errors.append("one or more package conversions are incomplete or failed")
        if not source_identity or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in source_identity.values()):
            provenance_errors.append("build source identity is empty or malformed")
        if "coreml_port/dynamic_queries.py" not in source_identity:
            provenance_errors.append("build source identity does not bind dynamic query policy")
        provenance_errors.extend(check_arithmetic_policy(args.arithmetic_policy, build, source_identity))
        seal_summary, seal_errors = verify_sealed_provenance(evidence_dir, build, manifest, source_identity)
        provenance_errors.extend(seal_errors)
        if not candidate_root.is_dir():
            provenance_errors.append("candidate package root is missing")
        current_acceptance = sha256(ROOT / "evidence/03-portable/ACCEPTANCE.json")
        if current_acceptance != FROZEN_ACCEPTANCE_SHA256:
            provenance_errors.append("repository frozen acceptance file differs from its pinned digest")
    except Exception as exc:
        manifest = {}; build = {}; source_identity = {}; digest = None
        provenance_errors.append(f"cannot load candidate provenance: {type(exc).__name__}: {exc}")
        current_acceptance = None

    current_adapter_sha256 = sha256(ROOT / "coreml_port/tiny_prefix_runtime.py")
    current_runtime_hashes = runtime_provenance()
    common = dict(candidate_root=candidate_root, package_digest=digest,
                  adapter=args.runtime_adapter, current_adapter_sha256=current_adapter_sha256,
                  current_runtime_provenance=current_runtime_hashes,
                  acceptance_sha256=FROZEN_ACCEPTANCE_SHA256)
    try:
        short_result = read_json(resolve_path(args.short_result))
        short_attempt = read_json(resolve_path(args.short_attempt))
        short = evaluate_short(short_result, short_attempt)
        short_errors = [*short["errors"], *check_common("short", {**short_result, "script_sha256": short_attempt.get("script_sha256"), "source_hashes": short_attempt.get("source_hashes", {})}, **common)]
        if short_attempt.get("runtime_provenance") != current_runtime_hashes:
            short_errors.append("short attempt runtime/guard identities are missing or stale")
        short.update(status="pass" if not short_errors and short["passed"] else "fail",
                     passed=not short_errors and short["passed"], result_path=str(resolve_path(args.short_result)),
                     attempt_path=str(resolve_path(args.short_attempt)), errors=short_errors)
    except Exception as exc:
        short = {"status": "invalid", "passed": False,
                 "errors": [f"cannot evaluate short evidence: {type(exc).__name__}: {exc}"]}

    gates = {
        "short": short,
        "partitions": load_gate("partitions", args.partitions_result, evaluate_partitions, **common),
        "boundaries": load_gate("boundaries", args.boundaries_result, evaluate_boundaries, **common),
        "generation": load_gate("generation", args.generation_result, evaluate_generation, **common),
    }
    provenance = {"status": "pass" if not provenance_errors else "fail",
                  "passed": not provenance_errors, "candidate_root": str(candidate_root),
                  "candidate_evidence_dir": str(evidence_dir), "manifest_path": str(manifest_path),
                  "package_aggregate_sha256": digest, "build_status": build.get("status"),
                  "source_files": sorted(source_identity), "seal": seal_summary, "errors": provenance_errors}
    missing = [name for name, gate in gates.items() if gate["status"] == "missing"]
    failed = (["provenance"] if not provenance["passed"] else []) + [name for name, gate in gates.items() if gate["status"] in {"fail", "invalid"} or not gate.get("passed")]
    all_complete = not missing and all(gate["status"] == "pass" for gate in gates.values())
    result = {
        "schema_version": 1,
        "status": "completed" if all_complete and provenance["passed"] else "incomplete" if missing else "completed_with_failures",
        "milestone_pass": provenance["passed"] and all_complete,
        "thresholds_changed": False,
        "acceptance_sha256": FROZEN_ACCEPTANCE_SHA256,
        "current_tiny_prefix_runtime_sha256": current_adapter_sha256,
        "current_runtime_provenance": current_runtime_hashes,
        "aggregator_sha256": sha256(Path(__file__)),
        "selected_runtime_adapter": args.runtime_adapter,
        "selected_arithmetic_policy": args.arithmetic_policy,
        "provenance": provenance,
        "gates": gates,
        "missing_gates": missing,
        "failed_gates": failed,
    }
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "milestone_pass": result["milestone_pass"],
                      "missing_gates": missing, "failed_gates": failed,
                      "output": str(output_path)}, indent=2), flush=True)
    return 0 if result["milestone_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
