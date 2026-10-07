"""Focused lifecycle tests for the opt-in tiny-prefix Core ML adapter."""

from __future__ import annotations

import unittest
import subprocess
import sys
from pathlib import Path

import numpy as np

from coreml_port.runtime import Pipeline, Request
from coreml_port.tiny_prefix_runtime import TinyPrefixPipeline, TinyPrefixRequest, select_pipeline

ROOT = Path(__file__).resolve().parents[1]


class FakeTinyPipeline(TinyPrefixPipeline):
    capacity = 2048
    valid_vocab = 1000

    def __init__(self):
        self.owner = object()
        self.created: list[Request] = []
        self.calls: list[list[int]] = []
        self.fail = False

    def _new_base_request(self):
        request = Request(self.owner, [object() for _ in range(12)])
        self.created.append(request)
        return request

    def _base_forward(self, ids, request, capture=None):
        self.calls.append(list(ids))
        for offset in range(0, len(ids), 32):
            chunk = ids[offset : offset + 32]
            if capture is not None:
                values = np.asarray(chunk, dtype=np.float32)[None, :, None]
                capture("final_hidden", values)
            request.length += len(chunk)
        if self.fail:
            raise KeyboardInterrupt("cancel fake forward")
        return np.asarray([[float(ids[-1])]], dtype=np.float32)


class TinyPrefixRuntimeTests(unittest.TestCase):
    def test_pipeline_selection_defaults_to_historical_runtime(self):
        self.assertIs(select_pipeline(), Pipeline)
        self.assertIs(select_pipeline(False), Pipeline)
        self.assertIs(select_pipeline(True), TinyPrefixPipeline)
        with self.assertRaises(ValueError):
            select_pipeline("yes")

    def test_candidate_identity_pairing_remains_required(self):
        with self.assertRaisesRegex(ValueError, "requires both"):
            TinyPrefixPipeline(model_dir=ROOT / "artifacts/not-loaded")
        with self.assertRaisesRegex(ValueError, "requires both"):
            Pipeline(model_dir=ROOT / "artifacts/not-loaded")

    def test_integrated_cli_flags_and_cpu_only_guards(self):
        commands = [
            [sys.executable, "-m", "coreml_port.generate", "--help"],
            [sys.executable, "scripts/validate_coreml.py", "--help"],
            [sys.executable, "scripts/validate_coreml_boundaries.py", "--help"],
            [sys.executable, "scripts/audit_coreml_partitions.py", "--help"],
            [sys.executable, "scripts/audit_coreml_generation.py", "--help"],
        ]
        for command in commands:
            result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
            self.assertIn("--tiny-prefix-replay", result.stdout)
        for command in [
            [sys.executable, "-m", "coreml_port.generate", "x", "--compute", "CPU_AND_GPU", "--tiny-prefix-replay"],
            [sys.executable, "scripts/validate_coreml.py", "--compute", "CPU_AND_GPU", "--tiny-prefix-replay"],
        ]:
            result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("requires --compute CPU_ONLY", result.stderr)

    def test_cpu_reference_warmup_rejects_other_compute_units(self):
        with self.assertRaisesRegex(ValueError, "requires CPU_ONLY"):
            TinyPrefixPipeline("CPU_AND_GPU")

    def test_recomputes_prefix_and_trims_capture_across_chunk_boundary(self):
        pipeline = FakeTinyPipeline()
        request = pipeline.request()
        emitted = []
        pipeline.forward(list(range(15)), request)
        self.assertEqual(request.warmup_ids, list(range(15)))

        pipeline.forward(list(range(100, 120)), request, lambda name, value: emitted.append((name, value.copy())))

        self.assertEqual(pipeline.calls, [list(range(15)), list(range(15)) + list(range(100, 120))])
        self.assertEqual(request.length, 35)
        self.assertEqual(request.warmup_ids, [])
        self.assertEqual(sum(value.shape[1] for _, value in emitted), 20)
        np.testing.assert_array_equal(np.concatenate([value for _, value in emitted], axis=1).reshape(-1), np.arange(100, 120))
        fresh = pipeline.created[-1]
        self.assertTrue(fresh.closed)
        self.assertFalse(fresh.states)
        self.assertEqual(len(request.states), 12)

    def test_overflow_is_rejected_before_state_or_history_mutation(self):
        pipeline = FakeTinyPipeline()
        request = pipeline.request()
        pipeline.forward([1, 2], request)
        old_states = request.states
        old_history = list(request.warmup_ids)
        request.length = pipeline.capacity
        with self.assertRaises(ValueError):
            pipeline.forward([3], request)
        self.assertIs(request.states, old_states)
        self.assertEqual(request.warmup_ids, old_history)
        self.assertEqual(len(pipeline.created), 1)

    def test_foreign_and_closed_requests_are_rejected(self):
        pipeline = FakeTinyPipeline()
        foreign = TinyPrefixRequest(object(), [object() for _ in range(12)])
        with self.assertRaises(ValueError):
            pipeline.forward([1], foreign)
        request = pipeline.request()
        request.close()
        self.assertEqual(request.warmup_ids, [])
        with self.assertRaises(ValueError):
            pipeline.forward([1], request)

    def test_cancellation_invalidates_original_and_disposes_fresh_request(self):
        pipeline = FakeTinyPipeline()
        request = pipeline.request()
        pipeline.forward([7, 8], request)
        old_states = request.states
        old_history = list(request.warmup_ids)
        pipeline.fail = True
        with self.assertRaises(KeyboardInterrupt):
            pipeline.forward([9], request)
        fresh = pipeline.created[-1]
        self.assertFalse(request.valid)
        self.assertIs(request.states, old_states)
        self.assertEqual(request.warmup_ids, old_history)
        self.assertTrue(fresh.closed)
        self.assertFalse(fresh.states)

    def test_history_is_bounded_and_cleared_at_warmup_boundary(self):
        pipeline = FakeTinyPipeline()
        request = pipeline.request()
        pipeline.forward(list(range(15)), request)
        pipeline.forward([15], request)
        self.assertEqual(request.length, 16)
        self.assertEqual(request.warmup_ids, [])
        pipeline.forward(list(range(16, 55)), request)
        self.assertEqual(request.length, 55)
        self.assertEqual(request.warmup_ids, [])


if __name__ == "__main__":
    unittest.main()
