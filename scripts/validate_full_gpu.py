"""Experimental full-model CPU_AND_GPU short gate on the original compiled cache.

This is a separate diagnostic runner. It does not alter the qualified loader,
thresholds, or runtime integration. It runs the frozen 54-comparison/17-check
short suite only; longer generation and extended qualification remain gated on
the outcome and are intentionally not part of this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coreml_port.tiny_prefix_runtime import TinyPrefixPipeline

EXPECTED_SOURCE_ID = "5a3e4c7a571e4fc3532377d9b9bb0da023f499c10709216fee7e1001bf4ad2c6"
EXPECTED_COMPILED_ID = "a838dd36e8f87c6b9fc0dd7865d0912651c6ab3399fa299d10ff12d788bcb031"
EXPECTED_ACCEPTANCE_SHA = "cb4d315266126df4a98194f73f8c10671310717c06ebfd191d3dac2334bf4233"
EXPECTED_INPUTS_SHA = "54ce299bdc107b3db80a2a389283ea787f7801a4f51a1ae2feed0a6dfaee7490"
SOURCE_ROOT = ROOT / "artifacts/04d-cpu/full-candidate-6"
SOURCE_MANIFEST = ROOT / "evidence/04d-cpu/full-candidate-6/package-identity.json"
COMPILED_ROOT = ROOT / "artifacts/05c-compiled/full-cache-1"
COMPILED_MANIFEST = ROOT / "evidence/05c-compiled/full-cache-1/compiled-identity.json"
MILESTONE = ROOT / "evidence/05c-compiled/milestone-rerun1.json"
ACCEPTANCE = ROOT / "evidence/03-portable/ACCEPTANCE.json"
INPUTS = ROOT / "evidence/01-reference/local-input-validation.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def provenance_sources() -> dict[str, str]:
    names = [
        "coreml_port/runtime.py", "coreml_port/tiny_prefix_runtime.py",
        "coreml_port/compiled_identity.py", "coreml_port/compiled_runtime.py",
        "coreml_port/loader_selection.py", "coreml_port/package_identity.py",
        "coreml_port/prediction_guard.py", "coreml_port/cleanup_guard.py",
        "portable/model.py", "coreml_port/generate.py", "scripts/probe_full_model.py",
        "scripts/runtime_provenance.py", "evidence/03-portable/ACCEPTANCE.json",
        "evidence/01-reference/local-input-validation.json",
    ]
    return {name: sha(ROOT / name) for name in names}


def verify_sealed_sources():
    from coreml_port.package_identity import verify_candidate_identity
    from coreml_port.compiled_identity import verify_compiled_identity
    source = verify_candidate_identity(SOURCE_ROOT, SOURCE_MANIFEST)
    compiled = verify_compiled_identity(COMPILED_ROOT, COMPILED_MANIFEST, SOURCE_ROOT, source)
    if source["aggregate_sha256"] != EXPECTED_SOURCE_ID or compiled["aggregate_sha256"] != EXPECTED_COMPILED_ID:
        raise RuntimeError("Selected source/compiled aggregate differs from the pinned qualified CPU candidate")
    return source, compiled


class ExperimentalGPUCompiledPipeline(TinyPrefixPipeline):
    """Isolated copy of the compiled/tiny-prefix adapter, with GPU permitted.

    Inference math and request/replay methods are inherited unchanged from the
    experimental Core ML pipeline. The ordinary persistent loader continues
    to reject non-CPU compute policies.
    """

    def __init__(self, compute, model_dir, identity_manifest, compiled_dir, compiled_manifest, on_model_loaded=None):
        import coremltools as ct
        from coreml_port.compiled_identity import verify_compiled_identity
        from coreml_port.package_identity import verify_candidate_identity, EXPECTED_PACKAGES
        from coreml_port.prediction_guard import install as install_prediction_guard
        from coreml_port.cleanup_guard import stats as cleanup_stats
        if compute != "CPU_AND_GPU":
            raise ValueError("This isolated experiment is pinned to CPU_AND_GPU")
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
        t = time.perf_counter()
        for name in EXPECTED_PACKAGES:
            spec = ct.models.utils.load_spec(str(self.model_dir / f"{name}.mlpackage"))
            if any(v.name == "norm_divisor" for v in spec.description.input):
                self.norm_divisor_inputs.add(name)
        self.source_spec_seconds = time.perf_counter() - t
        try:
            for name in EXPECTED_PACKAGES:
                t = time.perf_counter()
                self.models[name] = ct.models.CompiledMLModel(
                    str(self.compiled_dir / f"{name}.mlmodelc"),
                    compute_units=ct.ComputeUnit.CPU_AND_GPU)
                self.load_seconds[name] = time.perf_counter() - t
                if on_model_loaded is not None:
                    on_model_loaded(name, self.load_seconds[name])
        except BaseException:
            self.models.clear()
            raise
        self.initialization_seconds = time.perf_counter() - started
def install_experimental_pipeline_class():
    """Return the isolated adapter class; official loader remains untouched."""
    return ExperimentalGPUCompiledPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.attempt):
        parser.error("Attempt may contain only letters, digits, hyphens, and underscores")
    out = ROOT / "evidence/06b-full-gpu" / args.attempt
    out.mkdir(parents=True, exist_ok=False)
    protocol = ROOT / "evidence/06b-full-gpu/PROTOCOL.md"
    (out / "protocol-snapshot.md").write_bytes(protocol.read_bytes())
    (out / "source-snapshot.py").write_bytes(Path(__file__).read_bytes())
    result_path = out / "results.json"
    result = {"status": "starting", "experiment_completed": False,
              "compute_units": "CPU_AND_GPU", "actual_device_placement": "unverified",
              "runtime_adapter": "experimental_gpu_compiled_tiny_prefix_replay",
              "scope": "54 short numerical comparisons and 17 lifecycle checks only; no extended generation, partition, boundary, or performance qualification",
              "thresholds_changed": False, "comparisons": [], "checks": {}, "phases": []}

    def save():
        atomic_json(result_path, result)

    def phase(name, state):
        result["phases"].append({"phase": name, "state": state,
                                 "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        save()
        print(f"{name}: {state}", flush=True)

    save()
    cm = None
    try:
        import numpy as np
        import torch
        import coremltools as ct
        from portable.model import load_model
        from coreml_port.runtime import Request
        from coreml_port.generate import generate
        from scripts.probe_full_model import metrics
        from scripts.runtime_provenance import runtime_provenance
        from coreml_port.prediction_guard import install as install_prediction_guard, stats as prediction_stats
        from coreml_port.cleanup_guard import stats as cleanup_stats
        from coreml_port.loader_selection import loader_provenance as cpu_loader_provenance
        torch.set_num_threads(4)

        phase("preflight", "verifying pinned files, source and compiled identities")
        if sha(protocol) != sha(out / "protocol-snapshot.md") or sha(ACCEPTANCE) != EXPECTED_ACCEPTANCE_SHA or sha(INPUTS) != EXPECTED_INPUTS_SHA:
            raise RuntimeError("Pinned protocol, acceptance criteria, or input fixture changed")
        source_before, compiled_before = verify_sealed_sources()
        runtime_before = runtime_provenance()
        milestone = json.loads(MILESTONE.read_text())
        if milestone.get("milestone_pass") is not True or runtime_before != milestone.get("current_runtime_provenance"):
            raise RuntimeError("Current runtime/guards differ from the passing CPU compiled milestone")
        from argparse import Namespace
        source_args = Namespace(compiled_dir=str(COMPILED_ROOT), compiled_manifest=str(COMPILED_MANIFEST),
                                tiny_prefix_replay=True, compute="CPU_ONLY", model_dir=str(SOURCE_ROOT),
                                identity_manifest=str(SOURCE_MANIFEST))
        sealed_loader = cpu_loader_provenance(source_args)
        if sealed_loader != milestone["compiled_identity"]["loader_provenance"]:
            raise RuntimeError("Current compiled-loader source provenance differs from the passing CPU milestone")
        source_hashes_before = provenance_sources()
        # Install both native guards before any model constructors.
        prediction_initial = install_prediction_guard()
        cleanup_initial = cleanup_stats()
        if prediction_initial.get("installed") is not True:
            raise RuntimeError("Prediction guard did not install")
        prediction_counts_before = prediction_stats()
        result.update(source_identity=source_before["aggregate_sha256"],
                      compiled_identity=compiled_before["aggregate_sha256"],
                      compiled_manifest_sha256=sha(COMPILED_MANIFEST),
                      sealed_cpu_loader_provenance=sealed_loader,
                      experimental_adapter_source_sha256=sha(Path(__file__)),
                      runtime_provenance_before=runtime_before,
                      source_hashes_before=source_hashes_before,
                      prediction_guard_initial=prediction_initial,
                      prediction_guard_counts_before=prediction_counts_before,
                      cleanup_guard_initial=cleanup_initial,
                      input_fixture_sha256=EXPECTED_INPUTS_SHA,
                      acceptance_sha256=EXPECTED_ACCEPTANCE_SHA)
        result["script_sha256"] = sha(Path(__file__))
        result["protocol_sha256"] = sha(protocol)
        save()

        phase("model_load", "constructing isolated 14-package CPU_AND_GPU compiled pipeline")
        pipeline_type = install_experimental_pipeline_class()
        result["load_seconds"] = {}

        def record_model_load(name, elapsed):
            result["load_seconds"][name] = elapsed
            save()
            print(f"loaded {name} in {elapsed:.3f}s", flush=True)

        cm = pipeline_type("CPU_AND_GPU", SOURCE_ROOT, SOURCE_MANIFEST, COMPILED_ROOT,
                           COMPILED_MANIFEST, on_model_loaded=record_model_load)
        result["load_seconds"] = cm.load_seconds
        result["identity_verification_seconds"] = cm.identity_verification_seconds
        result["source_spec_seconds"] = cm.source_spec_seconds
        result["initialization_seconds"] = cm.initialization_seconds
        result["model_package_count"] = len(cm.models)
        if len(cm.models) != 14:
            raise RuntimeError(f"Expected 14 compiled models; loaded {len(cm.models)}")
        save()

        phase("short_suite", "running unchanged 54-comparison/17-check short gate")
        cases = json.loads(INPUTS.read_text())["cases"]
        if len(cases) != 3:
            raise RuntimeError(f"Expected three frozen short cases, got {len(cases)}")
        reference_model = load_model()

        def compare(name, got, want, score=False):
            value = metrics(torch.from_numpy(got), want)
            value["name"] = name
            if score:
                value["pass"] = value["max_abs"] <= .04
                value["top1_candidate"] = int(np.argmax(got.flatten()[:151667]))
                value["top1_reference"] = int(want.flatten()[:151667].argmax())
                value["top1_match"] = value["top1_candidate"] == value["top1_reference"]
                top = want.flatten()[:151667].topk(5)
                value["reference_margin"] = float(top.values[0] - top.values[1])
                value["top5_overlap"] = len(set(np.argsort(got.flatten()[:151667])[-5:].tolist()) & set(top.indices.tolist()))
                value["ranking_pass"] = value["top1_match"] or value["reference_margin"] <= .08
                value["pass"] = value["pass"] and value["ranking_pass"]
            else:
                value["pass"] = value["relative_rmse"] <= .01 and value["cosine"] >= .995
            result["comparisons"].append(value)

        with torch.inference_mode():
            for index, case in enumerate(cases):
                ids = case["input_ids"]
                captured, reference = {}, {}
                request = cm.request()

                def capture(name, value):
                    captured.setdefault(name, []).append(value.copy())

                got = cm.forward(ids, request, capture)
                hidden = reference_model(torch.tensor([ids]), capture=lambda name, value: reference.update({name: value.clone()}))
                want = reference_model.scores(hidden[:, -1:])
                compare(f"prompt{index}/scores", got, want, True)
                for name, parts in captured.items():
                    compare(f"prompt{index}/{name}", np.concatenate(parts, axis=1), reference[name])
                first = got.copy()
                for step in range(4):
                    token = int(want.flatten()[:151667].argmax())
                    ids = ids + [token]
                    got = cm.forward([token], request)
                    want = reference_model.scores(reference_model.reference_hidden(torch.tensor([ids]))[:, -1:])
                    compare(f"prompt{index}/teacher{step}", got, want, True)
                request.close()
                result["checks"][f"request{index}_disposed"] = request.closed and not request.states
                fresh = cm.request()
                again = cm.forward(case["input_ids"], fresh)
                result["checks"][f"request{index}_repeat_exact"] = bool(np.array_equal(first, again))
                fresh.close()
                save()
                print(f"short case {index} complete ({len(result['comparisons'])} comparisons)", flush=True)

        ids = cases[0]["input_ids"]
        request = cm.request()

        def reject(key, fn):
            try:
                fn()
                result["checks"][key] = False
            except (ValueError, RuntimeError):
                result["checks"][key] = True

        reject("padded_id_rejected", lambda: cm.forward([151667], request))
        reject("empty_rejected", lambda: cm.forward([], request))
        reject("foreign_owner_rejected", lambda: cm.forward([1], Request(object())))
        request.length = 2048
        reject("overflow_rejected", lambda: cm.forward([1], request))
        request.length = 0

        def cancel_capture(name, value):
            raise KeyboardInterrupt()
        try:
            cm.forward(ids, request, cancel_capture)
        except KeyboardInterrupt:
            result["checks"]["interrupt_invalidates"] = not request.valid
        reject("invalid_request_rejected", lambda: cm.forward([1], request))
        request.close()
        reject("closed_request_rejected", lambda: cm.forward([1], request))
        first_request = cm.request()
        first_token = int(cm.forward(ids, first_request).flatten()[:151667].argmax())
        first_request.close()
        seen = []
        generated = generate(cm, ids, 3, extra_stops=(first_token,), on_token=seen.append)
        result["checks"]["explicit_stop"] = generated["complete"] and len(generated["generated_ids"]) == 1
        result["checks"]["stream_callback"] = seen == generated["generated_ids"]
        result["stop_test"] = generated
        saved_request = cm.request
        owned = []

        def tracked_request():
            child = saved_request()
            owned.append(child)
            return child

        cm.request = tracked_request

        def interrupt_generation(token):
            raise KeyboardInterrupt()
        try:
            generate(cm, ids, 3, on_token=interrupt_generation)
        except KeyboardInterrupt:
            result["checks"]["generation_cancellation_disposes"] = owned[-1].closed and not owned[-1].states
        reject("generation_budget_rejected", lambda: generate(cm, ids, 2048))
        cm.request = saved_request

        result["cleanup_guard_final"] = cleanup_stats()
        result["prediction_guard_final"] = prediction_stats()
        pred_delta = {key: result["prediction_guard_final"][key] - prediction_counts_before[key]
                      for key in ("plain_calls", "state_calls", "gil_releases")}
        result["prediction_guard_delta"] = pred_delta
        result["prediction_guard_consistent"] = pred_delta["gil_releases"] == pred_delta["plain_calls"] + pred_delta["state_calls"]
        result["numerical_pass"] = len(result["comparisons"]) == 54 and all(c["pass"] for c in result["comparisons"])
        result["lifecycle_pass"] = len(result["checks"]) == 17 and all(result["checks"].values())
        result["comparison_count"] = len(result["comparisons"])
        result["check_count"] = len(result["checks"])
        result["first_failed_comparison"] = next((item for item in result["comparisons"]
                                                   if item.get("pass") is not True), None)
        result["thresholds"] = {"score_max_abs": 0.04, "hidden_relative_rmse_max": 0.01,
                                "hidden_cosine_min": 0.995, "score_ranking": "top1 match or reference margin <= 0.08",
                                "changed": False}
        result["actual_device_placement"] = "unverified; CPU_AND_GPU permits GPU but this run has no runtime device trace"
        result["operational_status"] = "completed"
        result["decision"] = "short numerical/lifecycle PASS" if result["numerical_pass"] and result["lifecycle_pass"] else "short numerical/lifecycle NO-GO; no extended GPU generation or qualification"
        phase("preservation", "rechecking source/cache/runtime and script inputs")
        source_after, compiled_after = verify_sealed_sources()
        runtime_after = runtime_provenance()
        source_hashes_after = provenance_sources()
        result["source_preserved"] = source_after["aggregate_sha256"] == EXPECTED_SOURCE_ID
        result["compiled_preserved"] = compiled_after["aggregate_sha256"] == EXPECTED_COMPILED_ID
        result["runtime_preserved"] = runtime_after == runtime_before
        result["source_hashes_preserved"] = source_hashes_after == source_hashes_before
        result["script_preserved"] = sha(Path(__file__)) == result["script_sha256"]
        result["protocol_preserved"] = sha(protocol) == result["protocol_sha256"]
        result["runtime_provenance_after"] = runtime_after
        result["source_hashes_after"] = source_hashes_after
        if not all((result["source_preserved"], result["compiled_preserved"], result["runtime_preserved"],
                    result["source_hashes_preserved"], result["script_preserved"],
                    result["protocol_preserved"], result["prediction_guard_consistent"])):
            raise RuntimeError("Preservation/provenance/guard counter check failed")
        result["experiment_completed"] = True
        result["status"] = "completed"
        result["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save()
        print(json.dumps({key: result[key] for key in ("status", "decision", "numerical_pass", "lifecycle_pass",
                                                       "comparison_count", "check_count", "actual_device_placement")},
                         indent=2), flush=True)
        return 0
    except BaseException as exc:
        result.update(status="failed", experiment_completed=False,
                      failure={"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
                      finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        save()
        print(f"failed: {type(exc).__name__}: {exc}", flush=True)
        return 1
    finally:
        if cm is not None:
            cm.models.clear()
        save()


if __name__ == "__main__":
    raise SystemExit(main())
