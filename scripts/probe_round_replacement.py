"""Qualify a v3 quotient-round replacement with an explicit full-Q limit.

The direct round output must match Torch byte-for-byte under both policies.
The full-Q output is separately compared to the frozen original Core ML CPU
output. Its strict Torch byte gate remains reported, including the already
observed negative-zero loss in the original Core ML quantizer.
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
BASELINE = ROOT / "evidence/06d-gpu-rounding/operator-baseline-1/results.json"
HELPER = ROOT / "scripts/signed_even_round_v3.py"
REFERENCE = ROOT / "coreml_port/graphs.py"
POLICIES = ("CPU_ONLY", "CPU_AND_GPU")
OUTPUTS = ("original_round", "candidate_round", "candidate_q")
EXTRA_LENGTHS = (1, 3, 12, 32)


def quotient(x):
    import torch

    exponent = torch.floor(torch.log2(torch.clamp(torch.abs(x), min=1e-30))) - 7
    powers = 2. ** torch.arange(-128, 128, dtype=torch.float32)
    step = powers[(exponent + 128).long()]
    return x / step


def extra_z(count):
    import numpy as np

    points = [np.float32(-0.0), np.float32(0.0)]
    for midpoint in (0.5, 1.5, 2.5):
        for sign in (-1, 1):
            value = np.float32(sign * midpoint)
            points.extend((np.nextafter(value, np.float32(-np.inf)), value,
                           np.nextafter(value, np.float32(np.inf))))
    vector = np.resize(np.asarray(points, dtype=np.float32), 256)
    rows = [np.roll(vector, 13 * index) for index in range(count)]
    return np.stack(rows, axis=0)[None, :, :].copy()


def original_cpu_output(baseline, name):
    import numpy as np

    rows = [row for row in baseline["cases"] if row["name"] == name]
    if len(rows) != 1:
        raise ValueError(f"Expected one original baseline case: {name}")
    row = rows[0]
    native = row["native"]["CPU_ONLY"]
    path = ROOT / native["output_archive"]
    if sha(path) != native["output_sha256"]:
        raise ValueError(f"Original CPU archive changed: {name}")
    with np.load(path) as stored:
        output = stored["actual"].copy()
    return row, output, native["output_sha256"]


def negative_zero_only(actual, expected):
    import numpy as np

    if actual.dtype != np.float32 or expected.dtype != np.float32 or actual.shape != expected.shape:
        return False, None
    a = np.ascontiguousarray(actual).view(np.uint32)
    e = np.ascontiguousarray(expected).view(np.uint32)
    changed = a != e
    return bool(np.all((e[changed] == 0x80000000) & (a[changed] == 0))), int(np.count_nonzero(changed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args()
    if args.attempt != "round-replacement-1":
        parser.error("This bounded gate permits only --attempt round-replacement-1")
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = read(protocol_path)
    source = Path(__file__)
    assert protocol["schema_version"] == 1 and protocol["attempt"] == args.attempt
    for key, path in (("script_sha256", source), ("helper_sha256", HELPER),
                      ("prior_result_sha256", PRIOR), ("baseline_result_sha256", BASELINE),
                      ("reference_sha256", REFERENCE)):
        assert protocol[key] == sha(path), key
    assert protocol["policies"] == list(POLICIES) and protocol["outputs"] == list(OUTPUTS)
    assert protocol["extra_lengths"] == list(EXTRA_LENGTHS)
    assert protocol["expected_cases"] == 24 and protocol["expected_native_predictions"] == 48
    evidence = ROOT / "evidence/06d-gpu-rounding" / args.attempt
    artifacts = ROOT / "artifacts/06d-gpu-rounding" / args.attempt
    if evidence.exists() or artifacts.exists():
        raise FileExistsError("Fresh replacement attempt required")
    evidence.mkdir(parents=True)
    artifacts.mkdir(parents=True)
    (evidence / "source-snapshot.py").write_bytes(source.read_bytes())
    (evidence / "helper-snapshot.py").write_bytes(HELPER.read_bytes())
    (evidence / "protocol.json").write_bytes(protocol_path.read_bytes())
    report = {"status": "starting", "passed": False,
              "script_sha256": sha(source), "helper_sha256": sha(HELPER),
              "prior_result_sha256": sha(PRIOR), "baseline_result_sha256": sha(BASELINE),
              "reference_sha256": sha(REFERENCE), "protocol_sha256": sha(protocol_path),
              "cases": [], "native_predictions": 0,
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
        from scripts.signed_even_round_v3 import Q_signed_even_v3, round_signed_even_v3

        torch.set_num_threads(4)
        prior = read(PRIOR)
        baseline = read(BASELINE)
        assert prior["status"] == baseline["status"] == "completed"
        assert len(prior["cases"]) == len(baseline["cases"]) == 20
        assert [r["name"] for r in prior["cases"]] == [r["name"] for r in baseline["cases"]]
        prepared = []
        baseline_negative_zero_changes = 0
        for old in prior["cases"]:
            path = ROOT / old["case_archive"]
            assert sha(path) == old["case_archive_sha256"]
            with np.load(path) as stored:
                x = stored["x"].copy()
                torch_q = stored["reference"].copy()
            assert x.dtype == np.float32 and x.shape[0] == 1 and x.shape[-1] == 256
            assert x.shape[1] in EXTRA_LENGTHS and np.isfinite(x).all()
            baseline_row, baseline_cpu, baseline_archive_sha = original_cpu_output(baseline, old["name"])
            assert old["input_sha256"] == baseline_row["input_sha256"] == hashlib.sha256(x.tobytes()).hexdigest()
            with torch.inference_mode():
                z = quotient(torch.from_numpy(x.copy())).numpy().copy()
                assert compare(Q(torch.from_numpy(x.copy())).numpy(), torch_q)["exact"]
            compatible, changed = negative_zero_only(baseline_cpu, torch_q)
            if not compatible:
                raise ValueError(f"Original CPU mismatch is not limited to negative zero: {old['name']}")
            baseline_negative_zero_changes += changed
            prepared.append((old["name"], x, z, torch_q, baseline_cpu, baseline_archive_sha, "prior quotient"))
        assert baseline_negative_zero_changes == protocol["baseline_cpu_negative_zero_changes"]
        for count in EXTRA_LENGTHS:
            base = next(item for item in prepared if item[0] == f"directed-q{count}")
            prepared.append((f"extra-half-q{count}", base[1].copy(), extra_z(count),
                             base[3].copy(), base[4].copy(), base[5],
                             "direct round ties, neighbors, and signed zeros"))
        assert len(prepared) == 24
        torch_failures = []
        for name, x, z, torch_q, baseline_cpu, baseline_archive_sha, origin in prepared:
            assert z.dtype == np.float32 and z.shape == x.shape and np.isfinite(z).all()
            with torch.inference_mode():
                tensor_z = torch.from_numpy(z.copy())
                tensor_x = torch.from_numpy(x.copy())
                round_ref = torch.round(tensor_z).numpy().copy()
                round_candidate = round_signed_even_v3(tensor_z).numpy().copy()
                q_candidate = Q_signed_even_v3(tensor_x).numpy().copy()
            checks = {"candidate_round_vs_torch": compare(round_candidate, round_ref),
                      "candidate_q_vs_torch": compare(q_candidate, torch_q)}
            if not all(item["exact"] for item in checks.values()):
                torch_failures.append(name)
            path = evidence / "cases" / f"{name}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, x=x, z=z, round_reference=round_ref,
                                torch_q=torch_q, baseline_cpu=baseline_cpu)
            report["cases"].append({"name": name, "count": int(x.shape[1]),
                                    "origin": origin, "case_archive": str(path.relative_to(ROOT)),
                                    "case_archive_sha256": sha(path),
                                    "baseline_cpu_archive_sha256": baseline_archive_sha,
                                    "torch_checks": checks, "native": {}})
            update()
        report["baseline_cpu_negative_zero_changes"] = baseline_negative_zero_changes
        report["torch_failures"] = torch_failures
        if torch_failures:
            raise AssertionError(f"Torch v3 failed sealed round/Q: {torch_failures}")

        class Replacement(nn.Module):
            def forward(self, x, z):
                return torch.round(z), round_signed_even_v3(z), Q_signed_even_v3(x)

        install_cleanup()
        install()
        assert stats() == {"installed": True, "plain_calls": 0,
                           "state_calls": 0, "gil_releases": 0}
        with torch.inference_mode():
            sample_x, sample_z = prepared[1][1:3]
            traced = torch.jit.trace(Replacement().eval(),
                                     (torch.from_numpy(sample_x.copy()),
                                      torch.from_numpy(sample_z.copy())), check_trace=False)
        started = time.perf_counter()
        converted = ct.convert(
            traced,
            inputs=[ct.TensorType(name="x", shape=(1, ct.RangeDim(1, 32, default=3), 256),
                                  dtype=np.float32),
                    ct.TensorType(name="z", shape=(1, ct.RangeDim(1, 32, default=3), 256),
                                  dtype=np.float32)],
            outputs=[ct.TensorType(name=name, dtype=np.float32) for name in OUTPUTS],
            convert_to="mlprogram", compute_precision=ct.precision.FLOAT32,
            minimum_deployment_target=ct.target.macOS15, skip_model_load=True)
        package = artifacts / "round-replacement.mlpackage"
        converted.save(str(package))
        report["conversion_seconds"] = time.perf_counter() - started
        report["package"] = str(package.relative_to(ROOT))
        report["package_files"] = inventory(package)
        update()
        compiled = artifacts / "round-replacement.mlmodelc"
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
                archive = ROOT / row["case_archive"]
                if sha(archive) != row["case_archive_sha256"]:
                    raise ValueError(f"Prepared case changed: {row['name']}")
                with np.load(archive) as stored:
                    x, z = stored["x"].copy(), stored["z"].copy()
                    round_ref = stored["round_reference"].copy()
                    torch_q = stored["torch_q"].copy()
                    baseline_cpu = stored["baseline_cpu"].copy()
                outputs = models[policy].predict({"x": x, "z": z})
                assert set(outputs) == set(OUTPUTS)
                actual = {key: outputs[key].copy() for key in OUTPUTS}
                checks = {
                    "original_round_vs_torch": compare(actual["original_round"], round_ref),
                    "candidate_round_vs_torch": compare(actual["candidate_round"], round_ref),
                    "candidate_q_vs_original_cpu": compare(actual["candidate_q"], baseline_cpu),
                    "candidate_q_vs_torch": compare(actual["candidate_q"], torch_q),
                }
                path = evidence / "native" / policy / f"{row['name']}.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, **actual)
                row["native"][policy] = {"checks": checks,
                                         "output_archive": str(path.relative_to(ROOT)),
                                         "output_archive_sha256": sha(path)}
                report["native_predictions"] += 1
                update()
            del models[policy]
            gc.collect()
        report["prediction_guard"] = stats()
        report["cleanup_guard"] = cleanup_stats()
        assert stats() == {"installed": True, "plain_calls": 48,
                           "state_calls": 0, "gil_releases": 48}
        assert inventory(package) == report["package_files"] and \
            inventory(compiled) == report["compiled_files"]
        for key, path in (("script_sha256", source), ("helper_sha256", HELPER),
                          ("prior_result_sha256", PRIOR), ("baseline_result_sha256", BASELINE),
                          ("reference_sha256", REFERENCE)):
            assert report[key] == sha(path), key
        assert report["protocol_sha256"] == sha(protocol_path)
        report["round_replacement_gate_passed"] = all(
            row["native"][policy]["checks"]["candidate_round_vs_torch"]["exact"]
            for row in report["cases"] for policy in POLICIES)
        report["original_cpu_round_control_passed"] = all(
            row["native"]["CPU_ONLY"]["checks"]["original_round_vs_torch"]["exact"]
            for row in report["cases"])
        report["native_cpu_full_q_preservation_passed"] = all(
            row["native"][policy]["checks"]["candidate_q_vs_original_cpu"]["exact"]
            for row in report["cases"] for policy in POLICIES)
        report["strict_full_q_torch_passed"] = all(
            row["native"][policy]["checks"]["candidate_q_vs_torch"]["exact"]
            for row in report["cases"] for policy in POLICIES)
        report["bounded_replacement_gate_passed"] = (report["original_cpu_round_control_passed"] and
                                                     report["round_replacement_gate_passed"] and
                                                     report["native_cpu_full_q_preservation_passed"])
        report.update(status="completed", passed=report["bounded_replacement_gate_passed"] and
                      report["strict_full_q_torch_passed"],
                      scope="Direct round and native original-CPU preservation are separate from the strict full-Q Torch gate; preexisting native negative-zero loss remains a no-go for that strict gate. CPU_AND_GPU does not establish device placement or composed-group fidelity.")
    except BaseException as exc:
        report.update(status="failed", passed=False,
                      error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        models.clear()
        gc.collect()
        update()
    print(json.dumps({"passed": report["passed"],
                      "original_cpu_round_control_passed": report["original_cpu_round_control_passed"],
                      "bounded_replacement_gate_passed": report["bounded_replacement_gate_passed"],
                      "strict_full_q_torch_passed": report["strict_full_q_torch_passed"],
                      "native_predictions": report["native_predictions"]}, indent=2))


if __name__ == "__main__":
    main()
