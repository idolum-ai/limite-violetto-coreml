"""Locate negative-zero loss and screen a magnitude/sign-factor round path.

This is an isolated diagnostic. It does not alter the frozen operator-1 model
or qualified group graphs. Root launches native execution after review.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.probe_signed_even_round import compare, inventory, read, save, sha

PRIOR = ROOT / "evidence/06d-gpu-rounding/operator-1/results.json"
PRIOR_SCRIPT = ROOT / "scripts/probe_signed_even_round.py"
HELPER = ROOT / "scripts/signed_even_round.py"
REFERENCE = ROOT / "coreml_port/graphs.py"
POLICIES = ("CPU_ONLY", "CPU_AND_GPU")
STAGES = ("quotient", "negative_branch", "signed_select",
          "zero_select", "denominator", "sign_factor", "rounded_magnitude",
          "old_final", "factor_final")


def values(x):
    """FP32 Torch stages for byte-exact reference and tracing."""
    import torch

    exponent = torch.floor(torch.log2(torch.clamp(torch.abs(x), min=1e-30))) - 7
    powers = 2. ** torch.arange(-128, 128, dtype=torch.float32)
    step = powers[(exponent + 128).long()]
    z = x / step
    magnitude = torch.abs(z)
    low = torch.floor(magnitude)
    fraction = magnitude - low
    odd = low - 2.0 * torch.floor(low / 2.0)
    advance = torch.where(fraction == 0.5, odd, (fraction > 0.5).to(z.dtype))
    rounded_magnitude = low + advance
    negative_branch = -rounded_magnitude
    signed_select = torch.where(z < 0.0, negative_branch, rounded_magnitude)
    zero_select = torch.where(z == 0.0, z, signed_select)
    denominator = torch.where(magnitude == 0.0, torch.ones_like(magnitude), magnitude)
    sign_factor = z / denominator
    old_final = zero_select * step
    factor_final = (rounded_magnitude * sign_factor) * step
    return (z, negative_branch, signed_select, zero_select,
            denominator, sign_factor, rounded_magnitude, old_final, factor_final)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args()
    if args.attempt != "zero-path-2":
        parser.error("This bounded diagnostic permits only --attempt zero-path-2")
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = read(protocol_path)
    source = Path(__file__)
    assert protocol["schema_version"] == 1 and protocol["attempt"] == args.attempt
    for key, path in (("script_sha256", source), ("prior_result_sha256", PRIOR),
                      ("prior_script_sha256", PRIOR_SCRIPT), ("helper_sha256", HELPER),
                      ("reference_sha256", REFERENCE)):
        assert protocol[key] == sha(path), key
    assert protocol["stages"] == list(STAGES)
    assert protocol["policies"] == list(POLICIES)
    assert protocol["expected_cases"] == 20
    evidence = ROOT / "evidence/06d-gpu-rounding" / args.attempt
    artifacts = ROOT / "artifacts/06d-gpu-rounding" / args.attempt
    if evidence.exists() or artifacts.exists():
        raise FileExistsError("Fresh diagnostic attempt required")
    evidence.mkdir(parents=True)
    artifacts.mkdir(parents=True)
    (evidence / "source-snapshot.py").write_bytes(source.read_bytes())
    (evidence / "protocol.json").write_bytes(protocol_path.read_bytes())
    report = {"status": "starting", "passed": False,
              "script_sha256": sha(source), "protocol_sha256": sha(protocol_path),
              "prior_result_sha256": sha(PRIOR), "prior_script_sha256": sha(PRIOR_SCRIPT),
              "helper_sha256": sha(HELPER), "reference_sha256": sha(REFERENCE),
              "stages": list(STAGES), "cases": [], "native_predictions": 0,
              "device_placement": "unverified; compute policy only"}

    def update():
        save(evidence / "results.json", report)

    update()
    models = {}
    try:
        import coremltools as ct
        import numpy as np
        import torch
        from torch import nn
        from coreml_port.graphs import Q
        from coreml_port.prediction_guard import install, stats
        from coreml_port.cleanup_guard import install as install_cleanup, stats as cleanup_stats

        torch.set_num_threads(4)
        prior = read(PRIOR)
        assert prior["status"] == "completed" and not prior["passed"]
        assert len(prior["cases"]) == protocol["expected_cases"]
        assert prior["script_sha256"] == sha(PRIOR_SCRIPT)
        torch_failures = []
        prepared = []
        for prior_case in prior["cases"]:
            archive = ROOT / prior_case["case_archive"]
            assert sha(archive) == prior_case["case_archive_sha256"]
            with np.load(archive) as raw:
                x = raw["x"].copy()
                reference = raw["reference"].copy()
            assert x.dtype == np.float32 and x.shape[0] == 1 and x.shape[-1] == 256
            assert x.shape[1] in (1, 3, 12, 32) and np.isfinite(x).all()
            with torch.inference_mode():
                stage_arrays = [a.numpy().copy() for a in values(torch.from_numpy(x.copy()))]
                sealed = Q(torch.from_numpy(x.copy())).numpy().copy()
            if not compare(sealed, reference)["exact"]:
                raise ValueError(f"Prior sealed reference mismatch: {prior_case['name']}")
            checks = {"old_final": compare(stage_arrays[7], reference),
                      "factor_final": compare(stage_arrays[8], reference)}
            if any(not c["exact"] for c in checks.values()):
                torch_failures.append(prior_case["name"])
            path = evidence / "torch" / f"{prior_case['name']}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, x=x, reference=reference,
                                **dict(zip(STAGES, stage_arrays)))
            row = {"name": prior_case["name"], "count": int(x.shape[1]),
                   "prior_archive_sha256": prior_case["case_archive_sha256"],
                   "torch_archive": str(path.relative_to(ROOT)),
                   "torch_archive_sha256": sha(path), "torch_checks": checks,
                   "native": {}}
            report["cases"].append(row)
            prepared.append(x)
            update()
        report["torch_failures"] = torch_failures
        if torch_failures:
            raise AssertionError(f"Torch stage/factor mismatch: {torch_failures}")

        class StageModel(nn.Module):
            def forward(self, x):
                return values(x)

        install_cleanup()
        install()
        assert stats() == {"installed": True, "plain_calls": 0,
                           "state_calls": 0, "gil_releases": 0}
        with torch.inference_mode():
            traced = torch.jit.trace(StageModel().eval(), torch.from_numpy(prepared[1].copy()),
                                     check_trace=False)
        started = time.perf_counter()
        converted = ct.convert(
            traced,
            inputs=[ct.TensorType(name="x", shape=(1, ct.RangeDim(1, 32, default=3), 256),
                                  dtype=np.float32)],
            outputs=[ct.TensorType(name=name, dtype=np.float32) for name in STAGES],
            convert_to="mlprogram", compute_precision=ct.precision.FLOAT32,
            minimum_deployment_target=ct.target.macOS15, skip_model_load=True)
        package = artifacts / "zero-path.mlpackage"
        converted.save(str(package))
        report["conversion_seconds"] = time.perf_counter() - started
        report["package"] = str(package.relative_to(ROOT))
        report["package_files"] = inventory(package)
        update()
        compiled = artifacts / "zero-path.mlmodelc"
        started = time.perf_counter()
        returned = ct.models.utils.compile_model(str(package), destination_path=str(compiled))
        assert Path(returned).resolve() == compiled.resolve()
        report["compile_seconds"] = time.perf_counter() - started
        report["compiled"] = str(compiled.relative_to(ROOT))
        report["compiled_files"] = inventory(compiled)
        update()

        for policy in POLICIES:
            unit = ct.ComputeUnit.CPU_ONLY if policy == "CPU_ONLY" else ct.ComputeUnit.CPU_AND_GPU
            models[policy] = ct.models.CompiledMLModel(str(compiled), compute_units=unit)
            for row in report["cases"]:
                archive = ROOT / row["torch_archive"]
                if sha(archive) != row["torch_archive_sha256"]:
                    raise ValueError(f"Torch stage archive changed: {row['name']}")
                with np.load(archive) as stored:
                    x = stored["x"].copy()
                    expected = {name: stored[name].copy() for name in STAGES}
                    reference = stored["reference"].copy()
                result = models[policy].predict({"x": x})
                assert set(result) == set(STAGES)
                actual = {name: result[name].copy() for name in STAGES}
                comparisons = {name: compare(actual[name], expected[name]) for name in STAGES}
                prior_native = prior_case_for(prior, row["name"])["native"][policy]
                prior_archive = ROOT / prior_native["output_archive"]
                assert sha(prior_archive) == prior_native["output_sha256"]
                with np.load(prior_archive) as old:
                    previous_final = old["actual"].copy()
                controls = {"old_vs_operator1": compare(actual["old_final"], previous_final),
                            "old_vs_sealed": compare(actual["old_final"], reference),
                            "factor_vs_sealed": compare(actual["factor_final"], reference)}
                path = evidence / "native" / policy / f"{row['name']}.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, **actual)
                row["native"][policy] = {"comparisons": comparisons, "controls": controls,
                                         "output_archive": str(path.relative_to(ROOT)),
                                         "output_archive_sha256": sha(path),
                                         "first_stage_mismatch": next((name for name in STAGES
                                                                       if not comparisons[name]["exact"]), None)}
                report["native_predictions"] += 1
                update()
            del models[policy]
            gc.collect()
        report["prediction_guard"] = stats()
        report["cleanup_guard"] = cleanup_stats()
        assert stats() == {"installed": True, "plain_calls": 40,
                           "state_calls": 0, "gil_releases": 40}
        assert inventory(package) == report["package_files"] and \
            inventory(compiled) == report["compiled_files"]
        for key, path in (("script_sha256", source), ("prior_result_sha256", PRIOR),
                          ("prior_script_sha256", PRIOR_SCRIPT), ("helper_sha256", HELPER),
                          ("reference_sha256", REFERENCE)):
            assert report[key] == sha(path), key
        assert report["protocol_sha256"] == sha(protocol_path)
        report["old_control_passed"] = all(
            row["native"][p]["controls"]["old_vs_operator1"]["exact"]
            for row in report["cases"] for p in POLICIES)
        report["factor_gate_passed"] = all(
            row["native"][p]["controls"]["factor_vs_sealed"]["exact"]
            for row in report["cases"] for p in POLICIES)
        report.update(status="completed", passed=report["old_control_passed"] and
                      report["factor_gate_passed"],
                      scope="Isolated FP32 diagnostic; output taps can change optimizer behavior, so old-control failure invalidates stage localization. GPU policy does not establish device placement.")
    except BaseException as exc:
        report.update(status="failed", passed=False,
                      error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        models.clear()
        gc.collect()
        update()
    print(json.dumps({"passed": report["passed"], "old_control_passed": report["old_control_passed"],
                      "factor_gate_passed": report["factor_gate_passed"],
                      "native_predictions": report["native_predictions"]}, indent=2))


def prior_case_for(prior, name):
    matches = [row for row in prior["cases"] if row["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"Expected one prior case: {name}")
    return matches[0]


if __name__ == "__main__":
    main()
