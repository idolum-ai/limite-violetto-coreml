"""Aggregate candidate-bound persistent compiled-loader qualification evidence.

This is a read-only evidence aggregator. It runs no model inference and never
changes the selected validator evidence or thresholds.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coreml_port.compiled_identity import verify_compiled_identity
from coreml_port.loader_selection import loader_provenance
from scripts import summarize_cpu_milestone as base


def check_loader_provenance(
    result: dict[str, Any], expected: dict[str, Any], package_digest: str,
) -> list[str]:
    """Require every gate to report the same current, candidate-bound loader."""
    errors: list[str] = []
    observed = result.get("loader_provenance")
    if observed != expected:
        errors.append("compiled loader provenance is missing, stale, or differs from selected manifest")
    if not isinstance(observed, dict):
        return errors
    if observed.get("mode") != "persistent_compiled":
        errors.append("loader mode is not persistent_compiled")
    if observed.get("source_candidate_identity") != package_digest:
        errors.append("compiled loader is not bound to the selected package identity")
    if observed.get("compiled_aggregate_sha256") != expected.get("compiled_aggregate_sha256"):
        errors.append("compiled aggregate identity differs from verified manifest")
    if observed.get("compiled_manifest_sha256") != expected.get("compiled_manifest_sha256"):
        errors.append("compiled manifest digest differs from current manifest")
    if observed.get("compiled_dir") != expected.get("compiled_dir") or observed.get("compiled_manifest") != expected.get("compiled_manifest"):
        errors.append("compiled paths differ from explicitly selected paths")
    if observed.get("sources") != expected.get("sources"):
        errors.append("compiled loader source identities are stale or incomplete")
    return errors


def load_compiled_gate(
    name: str, value: str, evaluate, *, common: dict[str, Any],
    expected_loader: dict[str, Any], package_digest: str,
) -> dict[str, Any]:
    gate = base.load_gate(name, value, evaluate, **common)
    if gate.get("status") in {"missing", "invalid"}:
        return gate
    try:
        result = base.read_json(base.resolve_path(value))
        errors = check_loader_provenance(result, expected_loader, package_digest)
        gate["errors"] = [*gate.get("errors", []), *errors]
        gate["passed"] = gate.get("passed") is True and not errors
        gate["status"] = "pass" if gate["passed"] else "fail"
    except Exception as exc:
        gate["passed"] = False
        gate["status"] = "invalid"
        gate.setdefault("errors", []).append(
            f"cannot verify compiled loader provenance: {type(exc).__name__}: {exc}"
        )
    return gate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", required=True, help="Explicit sealed 14-package source candidate root.")
    parser.add_argument("--candidate-evidence-dir", required=True, help="Source build, identity, and sealing evidence.")
    parser.add_argument("--compiled-dir", required=True, help="Persistent compiled directory containing all 14 models.")
    parser.add_argument("--compiled-manifest", required=True, help="Identity manifest for the persistent compiled directory.")
    parser.add_argument("--short-result", required=True)
    parser.add_argument("--short-attempt", required=True)
    parser.add_argument("--partitions-result", required=True)
    parser.add_argument("--boundaries-result", required=True)
    parser.add_argument("--generation-result", required=True)
    parser.add_argument("--arithmetic-policy", choices=sorted(base.ARITHMETIC_POLICIES), default="legacy-dynamic-product")
    parser.add_argument("--output", required=True, help="New summary JSON path; existing files are not overwritten.")
    args = parser.parse_args()

    candidate_root = base.resolve_path(args.candidate_root)
    evidence_dir = base.resolve_path(args.candidate_evidence_dir)
    compiled_dir = base.resolve_path(args.compiled_dir)
    compiled_manifest_path = base.resolve_path(args.compiled_manifest)
    output_path = base.resolve_path(args.output)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite compiled milestone summary: {output_path}")
    if not output_path.parent.is_dir():
        raise FileNotFoundError(f"Output parent directory must already exist: {output_path.parent}")

    errors: list[str] = []
    manifest_path = evidence_dir / "package-identity.json"
    build_path = evidence_dir / "build.json"
    source_path = evidence_dir / "source-identity.json"
    seal: dict[str, Any] = {}
    package_digest = None
    try:
        source_manifest = base.read_json(manifest_path)
        build = base.read_json(build_path)
        source_identity = base.read_json(source_path)
        package_digest = source_manifest.get("aggregate_sha256")
        names = [item.get("name") for item in source_manifest.get("packages", []) if isinstance(item, dict)]
        if names != list(base.EXPECTED_PACKAGES):
            errors.append("source package manifest does not list the ordered expected 14 packages")
        if not isinstance(package_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", package_digest):
            errors.append("source package aggregate identity is malformed")
        if build.get("identity") != package_digest:
            errors.append("source build identity differs from package manifest")
        if not isinstance(build.get("status"), str) or not build["status"].startswith("converted"):
            errors.append("source candidate build is not marked converted")
        if build.get("enhanced") is not True or build.get("source_exp") is not True or build.get("dynamic_product") is not True:
            errors.append("source candidate lacks qualified conversion options")
        copied = build.get("copied", {})
        conversions = build.get("conversions", [])
        described = list(copied) + [item.get("component") for item in conversions if isinstance(item, dict)]
        if set(described) != set(base.EXPECTED_PACKAGES) or len(described) != len(base.EXPECTED_PACKAGES):
            errors.append("source build does not describe each expected package exactly once")
        if not all(isinstance(item, dict) and item.get("exit_code") == 0 for item in conversions):
            errors.append("one or more source conversions are incomplete or failed")
        if not source_identity or any(not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v) for v in source_identity.values()):
            errors.append("source identity is empty or malformed")
        if "coreml_port/dynamic_queries.py" not in source_identity:
            errors.append("source identity does not bind dynamic query policy")
        errors.extend(base.check_arithmetic_policy(args.arithmetic_policy, build, source_identity))
        seal, seal_errors = base.verify_sealed_provenance(evidence_dir, build, source_manifest, source_identity)
        errors.extend(seal_errors)
        if not candidate_root.is_dir():
            errors.append("selected source candidate root is missing")
        if base.sha256(ROOT / "evidence/03-portable/ACCEPTANCE.json") != base.FROZEN_ACCEPTANCE_SHA256:
            errors.append("repository frozen acceptance file differs from pinned digest")
    except Exception as exc:
        source_manifest = {}; build = {}; source_identity = {}; package_digest = None
        errors.append(f"cannot load source candidate provenance: {type(exc).__name__}: {exc}")

    compiled_identity: dict[str, Any] = {}
    expected_loader: dict[str, Any] = {}
    compiled_identity_pass = False
    compiled_errors_before = len(errors)
    try:
        if not isinstance(package_digest, str):
            raise ValueError("source package identity is unavailable")
        compiled_identity = verify_compiled_identity(
            compiled_dir, compiled_manifest_path, candidate_root, source_manifest,
        )
        if compiled_identity.get("source_candidate_identity") != package_digest:
            errors.append("compiled manifest binds a different source package identity")
        if not isinstance(compiled_identity.get("aggregate_sha256"), str):
            errors.append("compiled manifest aggregate digest is missing")
        import argparse as _argparse
        loader_args = _argparse.Namespace(
            compiled_dir=str(compiled_dir), compiled_manifest=str(compiled_manifest_path),
            tiny_prefix_replay=True, compute="CPU_ONLY", model_dir=str(candidate_root),
            identity_manifest=str(manifest_path),
        )
        expected_loader = loader_provenance(loader_args)
        if expected_loader.get("source_candidate_identity") != package_digest:
            errors.append("loader metadata does not bind the selected source package identity")
        if expected_loader.get("compiled_aggregate_sha256") != compiled_identity.get("aggregate_sha256"):
            errors.append("loader metadata aggregate differs from fully verified compiled artifacts")
        compiled_identity_pass = len(errors) == compiled_errors_before
    except Exception as exc:
        errors.append(f"cannot verify persistent compiled identity: {type(exc).__name__}: {exc}")

    current_adapter_sha256 = base.sha256(ROOT / "coreml_port/tiny_prefix_runtime.py")
    current_runtime_hashes = base.runtime_provenance()
    common = dict(candidate_root=candidate_root, package_digest=package_digest,
                  adapter="compiled_tiny_prefix_replay", current_adapter_sha256=current_adapter_sha256,
                  current_runtime_provenance=current_runtime_hashes,
                  acceptance_sha256=base.FROZEN_ACCEPTANCE_SHA256)
    try:
        short_result = base.read_json(base.resolve_path(args.short_result))
        short_attempt = base.read_json(base.resolve_path(args.short_attempt))
        short = base.evaluate_short(short_result, short_attempt)
        short_errors = [*short["errors"], *base.check_common(
            "short", {**short_result, "script_sha256": short_attempt.get("script_sha256"),
                      "source_hashes": short_attempt.get("source_hashes", {})}, **common)]
        short_errors.extend(check_loader_provenance(short_result, expected_loader, package_digest or ""))
        short_errors.extend(check_loader_provenance(short_attempt, expected_loader, package_digest or ""))
        if short_attempt.get("runtime_adapter") != "compiled_tiny_prefix_replay":
            short_errors.append("short attempt runtime adapter is missing or differs from compiled loader")
        if short_attempt.get("runtime_provenance") != current_runtime_hashes:
            short_errors.append("short attempt runtime/guard identities are missing or stale")
        short.update(status="pass" if not short_errors and short["passed"] else "fail",
                     passed=not short_errors and short["passed"],
                     result_path=str(base.resolve_path(args.short_result)),
                     attempt_path=str(base.resolve_path(args.short_attempt)), errors=short_errors)
    except Exception as exc:
        short = {"status": "invalid", "passed": False,
                 "errors": [f"cannot evaluate short evidence: {type(exc).__name__}: {exc}"]}

    gates = {
        "short": short,
        "partitions": load_compiled_gate("partitions", args.partitions_result, base.evaluate_partitions,
                                          common=common, expected_loader=expected_loader, package_digest=package_digest or ""),
        "boundaries": load_compiled_gate("boundaries", args.boundaries_result, base.evaluate_boundaries,
                                          common=common, expected_loader=expected_loader, package_digest=package_digest or ""),
        "generation": load_compiled_gate("generation", args.generation_result, base.evaluate_generation,
                                          common=common, expected_loader=expected_loader, package_digest=package_digest or ""),
    }
    provenance = {"status": "pass" if not errors else "fail", "passed": not errors,
                  "candidate_root": str(candidate_root), "candidate_evidence_dir": str(evidence_dir),
                  "source_package_aggregate_sha256": package_digest, "seal": seal, "errors": errors}
    compiled = {"status": "pass" if compiled_identity_pass else "fail",
                "passed": compiled_identity_pass,
                "compiled_dir": str(compiled_dir), "compiled_manifest": str(compiled_manifest_path),
                "verified_identity": compiled_identity, "loader_provenance": expected_loader}
    missing = [name for name, gate in gates.items() if gate.get("status") == "missing"]
    failed = (["source_provenance"] if not provenance["passed"] else []) + (["compiled_identity"] if not compiled["passed"] else []) + [
        name for name, gate in gates.items() if gate.get("status") in {"fail", "invalid", "missing"} or not gate.get("passed")]
    all_complete = not missing and all(gate.get("status") == "pass" for gate in gates.values())
    passed = provenance["passed"] and compiled["passed"] and all_complete
    result = {"schema_version": 1, "status": "completed" if passed else "incomplete" if missing else "completed_with_failures",
              "milestone_pass": passed, "thresholds_changed": False,
              "acceptance_sha256": base.FROZEN_ACCEPTANCE_SHA256,
              "current_tiny_prefix_runtime_sha256": current_adapter_sha256,
              "current_runtime_provenance": current_runtime_hashes,
              "aggregator_sha256": base.sha256(Path(__file__)),
              "selected_runtime_adapter": "compiled_tiny_prefix_replay",
              "selected_arithmetic_policy": args.arithmetic_policy,
              "source_provenance": provenance, "compiled_identity": compiled,
              "gates": gates, "missing_gates": missing, "failed_gates": failed}
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "milestone_pass": passed,
                      "missing_gates": missing, "failed_gates": failed, "output": str(output_path)}, indent=2), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
