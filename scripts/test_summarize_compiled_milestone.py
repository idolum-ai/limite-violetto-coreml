"""Focused schema tests for the compiled-loader milestone aggregator."""
from __future__ import annotations

import unittest

from scripts.summarize_compiled_milestone import check_loader_provenance


class CompiledMilestoneTests(unittest.TestCase):
    def setUp(self):
        self.expected = {
            "mode": "persistent_compiled",
            "source_candidate_identity": "a" * 64,
            "compiled_aggregate_sha256": "b" * 64,
            "compiled_manifest_sha256": "c" * 64,
            "compiled_dir": "/sealed/compiled",
            "compiled_manifest": "/evidence/compiled-manifest.json",
            "sources": {
                "coreml_port/compiled_runtime.py": "d" * 64,
                "coreml_port/compiled_identity.py": "e" * 64,
                "coreml_port/loader_selection.py": "f" * 64,
                "coreml_port/package_identity.py": "0" * 64,
            },
        }

    def test_exact_candidate_bound_loader_identity_passes(self):
        self.assertEqual(check_loader_provenance(
            {"loader_provenance": self.expected}, self.expected, "a" * 64), [])

    def test_missing_or_stale_loader_identity_fails(self):
        self.assertTrue(check_loader_provenance({}, self.expected, "a" * 64))
        stale = {**self.expected, "compiled_aggregate_sha256": "9" * 64}
        errors = check_loader_provenance({"loader_provenance": stale}, self.expected, "a" * 64)
        self.assertTrue(errors)
        self.assertTrue(any("aggregate" in error for error in errors))

    def test_wrong_source_candidate_binding_fails(self):
        observed = {**self.expected, "source_candidate_identity": "9" * 64}
        errors = check_loader_provenance({"loader_provenance": observed}, self.expected, "a" * 64)
        self.assertTrue(errors)
        self.assertTrue(any("package identity" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
