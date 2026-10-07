"""Meaningful rejection tests for the 05e decode-product result summary."""
from __future__ import annotations

import unittest

from scripts.summarize_decode_product import (
    EXPECTED_GUARD,
    EXPECTED_OUTPUTS,
    EXPECTED_STATES,
    ROOT,
    TIMING_BOUNDARY,
    summarize_result,
)


SCRIPT_SHA = "a" * 64
PROTOCOL_SHA = "b" * 64
BASELINE_SHA = "c" * 64


def fixture(performance="go"):
    protocol = {
        "version": 1,
        "scope": "One-query group0, absolute positions32/128/512; no runtime integration",
        "build": "evidence/05e-product/fake-build.json",
        "build_sha256": "d" * 64,
        "fixtures": [{"results": f"evidence/fake/{p}.json", "sha256": digest * 64}
                     for p, digest in zip((32, 128, 512), "abc")],
        "script_sha256": SCRIPT_SHA,
        "go_median_reduction": 0.10,
        "cycles_per_block": 4,
        "blocks": 3,
    }
    def exact_call(prefix, arm, seconds=1.0):
        return {"prefix": prefix, "arm": arm, "seconds": seconds,
                "outputs_exact": {key: True for key in EXPECTED_OUTPUTS},
                "states_exact": {key: True for key in EXPECTED_STATES}, "passed": True}

    result = {
        "status": "completed", "experiment_completed": True,
        "script_sha256": SCRIPT_SHA, "protocol_sha256": PROTOCOL_SHA, "protocol": protocol,
        "timing_boundary": TIMING_BOUNDARY,
        "baseline_preservation": True, "baseline_identity": BASELINE_SHA,
        "runtime_provenance": {"runtime": "e" * 64},
        "prediction_guard": dict(EXPECTED_GUARD),
        "scope_rejections": 9, "exactness_passed": True,
        "correctness": [], "warmup": [], "timings": [], "blocks": [],
    }
    for prefix in (32, 128, 512):
        for repeat in range(2):
            for arm in "AB":
                row = exact_call(prefix, arm)
                row["repeat"] = repeat
                result["correctness"].append(row)
        for arm in "ABBAABBA":
            result["warmup"].append(exact_call(prefix, arm))

    orders = {0: (32, 128, 512), 1: (512, 32, 128), 2: (128, 512, 32)}
    variant_seconds = 0.80 if performance == "go" else 0.95
    for block, prefixes in orders.items():
        for prefix in prefixes:
            for cycle in range(4):
                for slot, arm in enumerate("ABBA"):
                    row = exact_call(prefix, arm, 1.0 if arm == "A" else variant_seconds)
                    row.update(block=block, cycle=cycle, slot=slot)
                    result["timings"].append(row)
    for block in range(3):
        for prefix in (32, 128, 512):
            reduction = 1.0 - variant_seconds
            result["blocks"].append({
                "block": block, "prefix": prefix,
                "baseline_median_seconds": 1.0,
                "variant_median_seconds": variant_seconds,
                "median_reduction": reduction,
                "go": reduction >= 0.10,
            })
    result["performance_gate_passed"] = performance == "go"
    result["decision"] = ("local performance go; full integration still unqualified"
                          if performance == "go" else "no-go at predeclared threshold; do not integrate")
    return result, protocol


class DecodeProductSummaryTests(unittest.TestCase):
    def summarize(self, result, protocol):
        return summarize_result(result, protocol, current_script_sha256=SCRIPT_SHA,
                                protocol_file_sha256=PROTOCOL_SHA)

    def test_valid_go_run_recomputes_all_nine_cells(self):
        result, protocol = fixture("go")
        summary = self.summarize(result, protocol)
        self.assertEqual(summary["status"], "completed")
        self.assertTrue(summary["experiment_valid"])
        self.assertTrue(summary["performance_gate_passed"])
        self.assertEqual(summary["correctness_call_count"], 12)
        self.assertEqual(summary["warmup_call_count"], 24)
        self.assertEqual(summary["timed_call_count"], 144)
        self.assertEqual(summary["block_count"], 9)

    def test_below_threshold_is_a_completed_experiment_not_failure(self):
        result, protocol = fixture("no-go")
        summary = self.summarize(result, protocol)
        self.assertEqual(summary["status"], "completed")
        self.assertTrue(summary["experiment_valid"])
        self.assertFalse(summary["performance_gate_passed"])
        self.assertIn("no-go", summary["decision"])

    def test_rejects_missing_timing_call(self):
        result, protocol = fixture()
        result["timings"].pop()
        self.assertFalse(self.summarize(result, protocol)["experiment_valid"])

    def test_rejects_duplicate_and_incorrect_arm_or_order(self):
        result, protocol = fixture()
        result["correctness"][1] = dict(result["correctness"][0])
        self.assertFalse(self.summarize(result, protocol)["experiment_valid"])

        result, protocol = fixture()
        result["timings"][0]["arm"] = "B"
        self.assertFalse(self.summarize(result, protocol)["experiment_valid"])

        result, protocol = fixture()
        result["timings"][0], result["timings"][1] = result["timings"][1], result["timings"][0]
        self.assertFalse(self.summarize(result, protocol)["experiment_valid"])

    def test_rejects_inexact_output_or_missing_state_entry(self):
        result, protocol = fixture()
        result["correctness"][0]["outputs_exact"]["y"] = False
        self.assertFalse(self.summarize(result, protocol)["experiment_valid"])

        result, protocol = fixture()
        result["warmup"][0]["states_exact"].pop(next(iter(EXPECTED_STATES)))
        self.assertFalse(self.summarize(result, protocol)["experiment_valid"])

    def test_rejects_failed_baseline_guard_and_scope_gates(self):
        for field, value in (("baseline_preservation", False),
                             ("scope_rejections", 8),
                             ("prediction_guard", {**EXPECTED_GUARD, "state_calls": 179})):
            result, protocol = fixture()
            result[field] = value
            self.assertFalse(self.summarize(result, protocol)["experiment_valid"], field)

    def test_rejects_tampered_block_median_or_decision(self):
        result, protocol = fixture()
        result["blocks"][0]["median_reduction"] = 0.99
        self.assertFalse(self.summarize(result, protocol)["experiment_valid"])

        result, protocol = fixture("no-go")
        result["decision"] = "local performance go; full integration still unqualified"
        self.assertFalse(self.summarize(result, protocol)["experiment_valid"])

    def test_rejects_protocol_or_script_identity_change(self):
        result, protocol = fixture()
        protocol["go_median_reduction"] = 0.05
        result["protocol"] = protocol
        self.assertFalse(self.summarize(result, protocol)["experiment_valid"])

        result, protocol = fixture()
        summary = summarize_result(result, protocol, current_script_sha256="f" * 64,
                                  protocol_file_sha256=PROTOCOL_SHA)
        self.assertFalse(summary["experiment_valid"])


if __name__ == "__main__":
    unittest.main()
