"""Schema checks for the machine-readable candidate milestone aggregator."""

from __future__ import annotations

import unittest
import hashlib
import json
import tempfile
from pathlib import Path

from scripts.summarize_cpu_milestone import (
    EXPECTED_BOUNDARY_LABELS,
    EXPECTED_PARTITION_CASES,
    evaluate_boundaries,
    evaluate_generation,
    evaluate_partitions,
    evaluate_short,
    check_common,
    check_arithmetic_policy,
    load_gate,
    verify_sealed_provenance,
    sha256,
    FROZEN_ACCEPTANCE_SHA256,
)


class MilestoneSummaryTests(unittest.TestCase):
    def test_legacy_policy_accepts_candidate4_build_without_candidate5_flags(self):
        build = {"dynamic_product": True}
        source_identity = {"coreml_port/dynamic_queries.py": "a" * 64}
        self.assertEqual(check_arithmetic_policy("legacy-dynamic-product", build, source_identity), [])

    def test_paired_tiny_policy_requires_flags_helpers_and_every_group_command(self):
        helpers = {
            "coreml_port/paired_product.py": "a" * 64,
            "coreml_port/tiny_scores.py": "b" * 64,
        }
        command = ["python", "convert_coreml.py", "--product", "paired",
                   "--score-dot", "tiny-sequential"]
        build = {"paired_product": True, "tiny_sequential": True,
                 "conversions": [{"component": str(index), "command": command}
                                 for index in range(12)]}
        self.assertEqual(check_arithmetic_policy("paired-tiny", build, helpers), [])

        missing_flags = dict(build, paired_product=False, tiny_sequential=False)
        errors = check_arithmetic_policy("paired-tiny", missing_flags, helpers)
        self.assertTrue(any("paired_product=true" in error for error in errors))
        self.assertTrue(any("tiny_sequential=true" in error for error in errors))
        errors = check_arithmetic_policy("paired-tiny", {"conversions": build["conversions"]}, helpers)
        self.assertTrue(any("paired_product=true" in error for error in errors))
        self.assertTrue(any("tiny_sequential=true" in error for error in errors))

        errors = check_arithmetic_policy("paired-tiny", build, {})
        self.assertTrue(any("paired_product.py" in error for error in errors))
        self.assertTrue(any("tiny_scores.py" in error for error in errors))

        partial = [{"component": str(index), "command": command} for index in range(12)]
        partial[8] = {"component": "8", "command": ["python", "convert_coreml.py"]}
        errors = check_arithmetic_policy("paired-tiny", dict(build, conversions=partial), helpers)
        self.assertTrue(any("group 8" in error and "--product paired" in error for error in errors))
        self.assertTrue(any("group 8" in error and "--score-dot tiny-sequential" in error for error in errors))
        missing_group = [item for item in build["conversions"] if item["component"] != "11"]
        errors = check_arithmetic_policy("paired-tiny", dict(build, conversions=missing_group), helpers)
        self.assertTrue(any("missing groups" in error and "11" in error for error in errors))

        errors = check_arithmetic_policy("paired-tiny", dict(build, tiny_native=True), helpers)
        self.assertTrue(any("forbids build tiny_native=true" in error for error in errors))
        native_command = ["python", "convert_coreml.py", "--product", "paired",
                          "--score-dot", "tiny-native"]
        native_conversions = [{"component": str(index), "command": native_command}
                              for index in range(12)]
        errors = check_arithmetic_policy("paired-tiny", dict(build, conversions=native_conversions), helpers)
        self.assertTrue(any("lacks --score-dot tiny-sequential" in error for error in errors))
        self.assertTrue(any("forbids --score-dot tiny-native" in error for error in errors))

    def test_paired_product_policy_accepts_capacity_only_mode_and_rejects_tiny_score_mode(self):
        command = ["python", "convert_coreml.py", "--product", "paired"]
        build = {"paired_product": True, "tiny_sequential": False,
                 "conversions": [{"component": str(index), "command": command}
                                 for index in range(12)]}
        identity = {"coreml_port/paired_product.py": "a" * 64}
        self.assertEqual(check_arithmetic_policy("paired-product", build, identity), [])

        errors = check_arithmetic_policy("paired-product", {**build, "paired_product": False}, identity)
        self.assertTrue(any("paired_product=true" in error for error in errors))
        errors = check_arithmetic_policy("paired-product", {**build, "tiny_sequential": True}, identity)
        self.assertTrue(any("forbids build tiny_sequential=true" in error for error in errors))

        tiny_command = [*command, "--score-dot", "tiny-sequential"]
        conversions = [{"component": str(index), "command": tiny_command}
                       for index in range(12)]
        errors = check_arithmetic_policy("paired-product", {**build, "conversions": conversions}, identity)
        self.assertTrue(any("forbids --score-dot tiny-sequential" in error for error in errors))

        errors = check_arithmetic_policy("paired-product", build, {})
        self.assertTrue(any("paired_product.py" in error for error in errors))

        native_build = {**build, "tiny_native": True}
        errors = check_arithmetic_policy("paired-product", native_build, identity)
        self.assertTrue(any("forbids build tiny_native=true" in error for error in errors))
        native_command = [*command, "--score-dot", "tiny-native"]
        conversions = [{"component": str(index), "command": native_command}
                       for index in range(12)]
        errors = check_arithmetic_policy("paired-product", {**build, "conversions": conversions}, identity)
        self.assertTrue(any("forbids --score-dot tiny-native" in error for error in errors))

    def test_paired_native_tiny_policy_requires_full_seal_identity_and_commands(self):
        helpers = {
            "coreml_port/paired_product.py": "a" * 64,
            "coreml_port/tiny_native_scores.py": "b" * 64,
        }
        command = ["python", "convert_coreml.py", "--product", "paired",
                   "--score-dot", "tiny-native"]
        build = {
            "paired_product": True, "tiny_native": True, "tiny_sequential": False,
            "enhanced": True, "source_exp": True, "dynamic_product": True,
            "conversions": [{"component": str(index), "command": command} for index in range(12)],
        }
        self.assertEqual(check_arithmetic_policy("paired-native-tiny", build, helpers), [])

        for option, value in (("paired_product", False), ("tiny_native", False),
                              ("tiny_sequential", True), ("enhanced", False),
                              ("source_exp", False), ("dynamic_product", False)):
            errors = check_arithmetic_policy("paired-native-tiny", {**build, option: value}, helpers)
            self.assertTrue(errors, f"expected {option}={value!r} to fail")
        errors = check_arithmetic_policy("paired-native-tiny", build, {})
        self.assertTrue(any("paired_product.py" in error for error in errors))
        self.assertTrue(any("tiny_native_scores.py" in error for error in errors))

        wrong_dot = ["python", "convert_coreml.py", "--product", "paired",
                     "--score-dot", "tiny-sequential"]
        wrong_conversions = [{"component": str(index), "command": wrong_dot} for index in range(12)]
        errors = check_arithmetic_policy("paired-native-tiny", {**build, "conversions": wrong_conversions}, helpers)
        self.assertTrue(any("lacks --score-dot tiny-native" in error for error in errors))
        self.assertTrue(any("forbids --score-dot tiny-sequential" in error for error in errors))

        missing_product = ["python", "convert_coreml.py", "--score-dot", "tiny-native"]
        missing_product_commands = [{"component": str(index), "command": missing_product}
                                    for index in range(12)]
        errors = check_arithmetic_policy("paired-native-tiny", {**build, "conversions": missing_product_commands}, helpers)
        self.assertTrue(any("lacks --product paired" in error for error in errors))
        missing_group = [item for item in build["conversions"] if item["component"] != "11"]
        errors = check_arithmetic_policy("paired-native-tiny", {**build, "conversions": missing_group}, helpers)
        self.assertTrue(any("missing groups" in error and "11" in error for error in errors))

        legacy_native = check_arithmetic_policy(
            "legacy-dynamic-product", {"dynamic_product": True, "tiny_native": True}, {})
        self.assertTrue(legacy_native)
        legacy_native_command = check_arithmetic_policy(
            "legacy-dynamic-product", {"dynamic_product": True, "conversions": [
                {"component": "0", "command": command}]}, {})
        self.assertTrue(any("forbids --score-dot tiny-native" in error for error in legacy_native_command))

    def test_short_gate_requires_exact_54_and_17_pass_counts(self):
        result = {
            "comparisons": [{"pass": True, "max_abs": 1e-6, "relative_rmse": 0.0, "name": "prompt0/scores"}] * 54,
            "checks": {f"check{i}": True for i in range(17)},
            "numerical_pass": True,
            "lifecycle_pass": True,
            "thresholds_changed": False,
        }
        attempt = {"status": "completed"}
        summary = evaluate_short(result, attempt)
        self.assertTrue(summary["passed"])
        result["comparisons"].pop()
        self.assertFalse(evaluate_short(result, attempt)["passed"])

    def test_boundaries_require_exact_label_set_and_lifecycle_checks(self):
        result = {
            "status": "completed",
            "comparisons": [{"name": name, "pass": True, "scores": {"max_abs": 1e-6}, "hidden": {"relative_rmse": 0.0}} for name in EXPECTED_BOUNDARY_LABELS],
            "checks": {f"check{i}": True for i in range(4)},
            "numerical_pass": True,
            "checks_pass": True,
        }
        self.assertTrue(evaluate_boundaries(result)["passed"])
        result["checks"]["check0"] = False
        self.assertFalse(evaluate_boundaries(result)["passed"])

    def test_partitions_require_all_97_prefixes_disposal_and_repeat(self):
        cases = []
        for name, steps in EXPECTED_PARTITION_CASES.items():
            cases.append({
                "name": name,
                "passed": True,
                "request_disposed": True,
                "matched_prefixes": [
                    {"passed": True, "scores": {"max_abs": 1e-6}, "hidden": {"relative_rmse": 0.0}}
                    for _ in range(steps)
                ],
            })
        result = {
            "status": "completed", "cases": cases, "numerical_pass": True, "passed": True,
            "partition_repeat_pass": True,
            "partition_repeat": {"scores_exact": True, "hidden_exact": True, "requests_disposed": True},
        }
        summary = evaluate_partitions(result)
        self.assertTrue(summary["passed"])
        self.assertEqual(summary["matched_prefix_count"], 97)
        result["cases"].pop()
        self.assertFalse(evaluate_partitions(result)["passed"])

    def test_generation_requires_periodic_and_terminal_audits(self):
        result = {
            "status": "completed", "audits_pass": True, "generation_pass": True, "passed": True,
            "audits": [
                {"label": "periodic_0", "scores": {"passed": True}},
                {"label": "periodic_128", "scores": {"passed": True}},
                {"label": "terminal", "scores": {"passed": True}},
            ],
            "generation": {
                "generated_tokens": 129, "finish_reason": "eos", "complete": True,
                "boxed_391_after_think": True, "think_marker_present": True,
                "qualified_reference_settings_match": True, "request_disposed": True,
            },
        }
        self.assertTrue(evaluate_generation(result)["passed"])
        result["audits"].pop(1)
        self.assertFalse(evaluate_generation(result)["passed"])

    def test_absent_gate_is_explicitly_missing_and_failing(self):
        gate = load_gate("partitions", None, evaluate_partitions)
        self.assertEqual(gate["status"], "missing")
        self.assertFalse(gate["passed"])

    def test_gate_rejects_missing_or_mixed_runtime_guard_identity(self):
        import scripts.summarize_cpu_milestone as aggregator
        # This tests identity binding, not the availability of a locally built
        # native guard. Synthetic hashes keep the test runnable from a clone.
        runtime = {
            "coreml_port/runtime.py": "b" * 64,
            "artifacts/04b-reliability/cleanup_guard.dylib": "c" * 64,
            "artifacts/04e-runtime/prediction_guard.dylib": "d" * 64,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            result = {
                "package_identity": {"aggregate_sha256": "a" * 64},
                "model_dir": str(root), "runtime_adapter": "tiny_prefix_replay",
                "tiny_prefix_runtime_sha256": sha256(aggregator.ROOT / "coreml_port/tiny_prefix_runtime.py"),
                "acceptance_sha256": FROZEN_ACCEPTANCE_SHA256,
                "thresholds_changed": False, "compute": "CPU_ONLY",
                "validator_sha256": sha256(aggregator.ROOT / "scripts/audit_coreml_partitions.py"),
                "runtime_provenance": runtime,
            }
            common = dict(candidate_root=root, package_digest="a" * 64,
                          adapter="tiny_prefix_replay",
                          current_adapter_sha256=result["tiny_prefix_runtime_sha256"],
                          current_runtime_provenance=runtime,
                          acceptance_sha256=FROZEN_ACCEPTANCE_SHA256)
            self.assertEqual(check_common("partitions", result, **common), [])
            altered = dict(result, runtime_provenance={**runtime, "coreml_port/runtime.py": "0" * 64})
            errors = check_common("partitions", altered, **common)
            self.assertTrue(any("runtime/cleanup/prediction guard identities" in error for error in errors))

    def test_provenance_rejects_failed_seal_or_copied_lineage(self):
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary)
            snapshot = evidence / "source-snapshots/coreml_port/example.py"
            snapshot.parent.mkdir(parents=True)
            snapshot.write_text("sealed source\n")
            source_digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
            identity = {"coreml_port/example.py": source_digest}
            aggregate = "a" * 64
            copied = [{
                "component": "head", "source_package": "artifacts/old/head.mlpackage",
                "same_file_set": True, "pass": True,
                "files": [{"size_matches": True, "sha256": "b" * 64, "expected": "b" * 64}],
            }]
            seal = {
                "pass": True, "sources_unchanged_during_build": True,
                "before": identity, "after": identity,
                "package_identity": aggregate, "copied_lineage": copied,
            }
            seal_path = evidence / "provenance-verification.json"
            seal_path.write_text(json.dumps(seal))
            build = {"copied": {"head": "artifacts/old/head.mlpackage"}}
            manifest = {"aggregate_sha256": aggregate}
            _, errors = verify_sealed_provenance(evidence, build, manifest, identity)
            self.assertEqual(errors, [])

            seal["sources_unchanged_during_build"] = False
            seal["pass"] = False
            seal["copied_lineage"][0]["pass"] = False
            seal_path.write_text(json.dumps(seal))
            _, errors = verify_sealed_provenance(evidence, build, manifest, identity)
            self.assertTrue(any("pass flag is false" in error for error in errors))
            self.assertTrue(any("sources changed" in error for error in errors))
            self.assertTrue(any("copied lineage did not pass" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
