"""Exercise the publication scanner against real staged blobs, without models."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

AUDITOR = Path(__file__).resolve().with_name("audit_git_index.py")


class PublicationAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="violetto-publication-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        subprocess.run(["git", "init", "-q", "--initial-branch=main", str(self.root)], check=True)

    def stage(self, path, text):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        subprocess.run(["git", "add", "--", path], cwd=self.root, check=True)
        return target

    def audit(self):
        result = subprocess.run([sys.executable, str(AUDITOR)], cwd=self.root,
                                text=True, capture_output=True)
        self.assertIn(result.returncode, (0, 1), result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(result.returncode == 0, report["passed"])
        return result, report

    def test_preserves_redacted_power_and_library_build_identity(self):
        self.stage("evidence/results.json", json.dumps({
            "power": "Battery Power; InternalBattery-0 (id=<REDACTED_HOST_ID>); 90%",
            "library_build_uuid": "284A8C1E-8591-3376-8FC6-C462FD9416C3",
            "score_check_passed": False,
        }))
        _, report = self.audit()
        self.assertTrue(report["passed"])

    def test_checks_index_even_after_working_tree_is_redacted(self):
        identifier = "123456789"
        target = self.stage("evidence/results.json", json.dumps({
            "power": "InternalBattery-0 (id=" + identifier + ")",
        }))
        target.write_text('{"power": "Battery Power"}\n')
        result, report = self.audit()
        self.assertFalse(report["passed"])
        self.assertEqual(report["failures"][0]["rule"], "host-local battery identifier")
        self.assertNotIn(identifier, result.stdout)

    def test_rejects_private_trace_entries_but_keeps_model_hashes(self):
        digest = "a" * 64
        self.stage("evidence/SHA256SUMS", digest + "  artifacts/model.mil\n")
        _, report = self.audit()
        self.assertTrue(report["passed"])
        self.stage("evidence/SHA256SUMS", digest + "  evidence/run.trace/data/1.zip\n")
        _, report = self.audit()
        self.assertFalse(report["passed"])
        self.assertIn("private trace checksum entry", [x["rule"] for x in report["failures"]])

    def test_rejects_trace_artifacts_and_private_history_bundles(self):
        self.stage("evidence/run.trace/metadata.txt", "private trace metadata\n")
        self.stage("private-history.bundle", "not an actual bundle\n")
        _, report = self.audit()
        self.assertFalse(report["passed"])
        self.assertEqual({x["path"] for x in report["failures"]},
                         {"evidence/run.trace/metadata.txt", "private-history.bundle"})


if __name__ == "__main__":
    unittest.main()
