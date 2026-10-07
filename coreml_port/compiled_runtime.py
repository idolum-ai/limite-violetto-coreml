"""Opt-in persistent compiled loading; inherited inference math is unchanged."""
from __future__ import annotations
from pathlib import Path
import time
import coremltools as ct
from .tiny_prefix_runtime import TinyPrefixPipeline
from .package_identity import verify_candidate_identity, EXPECTED_PACKAGES
from .compiled_identity import verify_compiled_identity
from .prediction_guard import install as install_prediction_guard
from .cleanup_guard import stats as cleanup_stats

class CompiledTinyPrefixPipeline(TinyPrefixPipeline):
    runtime_adapter = 'compiled_tiny_prefix_replay'

    def __init__(self, compute='CPU_ONLY', model_dir=None, identity_manifest=None, *, compiled_dir, compiled_manifest):
        if compute != 'CPU_ONLY':
            raise ValueError('Persistent compiled loading requires CPU_ONLY.')
        if model_dir is None or identity_manifest is None:
            raise ValueError('Persistent compiled loading requires explicit sealed source selection.')
        started = time.perf_counter()
        self.model_dir = Path(model_dir).expanduser()
        self.package_identity = verify_candidate_identity(self.model_dir, identity_manifest)
        self.model_dir = self.model_dir.resolve()
        self.compiled_dir = Path(compiled_dir).expanduser()
        self.compiled_identity = verify_compiled_identity(self.compiled_dir, compiled_manifest,
                                                          self.model_dir, self.package_identity)
        self.compiled_dir = self.compiled_dir.resolve()
        self.identity_verification_seconds = time.perf_counter() - started
        self.prediction_guard = install_prediction_guard()
        self.cleanup_guard = cleanup_stats()
        self.owner = object()
        self.compute = compute
        self.load_seconds = {}
        self.models = {}
        self.norm_divisor_inputs = set()
        # Read source specs outside model construction: CompiledMLModel has no get_spec.
        start = time.perf_counter()
        for name in EXPECTED_PACKAGES:
            spec = ct.models.utils.load_spec(str(self.model_dir / f'{name}.mlpackage'))
            if any(v.name == 'norm_divisor' for v in spec.description.input):
                self.norm_divisor_inputs.add(name)
        self.source_spec_seconds = time.perf_counter() - start
        try:
            for name in EXPECTED_PACKAGES:
                start = time.perf_counter()
                self.models[name] = ct.models.CompiledMLModel(str(self.compiled_dir / f'{name}.mlmodelc'),
                                                            compute_units=ct.ComputeUnit.CPU_ONLY)
                self.load_seconds[name] = time.perf_counter() - start
        except BaseException:
            self.models.clear()
            raise
        self.initialization_seconds = time.perf_counter() - started
