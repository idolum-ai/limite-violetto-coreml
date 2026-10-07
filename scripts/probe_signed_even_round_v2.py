"""Bounded v2 sign-factor quantizer qualification on directed and captured values.

The sealed `graphs.Q` is the reference. Torch is checked first, then one
side-by-side Core ML operator is tested under CPU_ONLY and CPU_AND_GPU. Every
case is reported; a mismatch never stops later cases from being measured.
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
SOURCE_RESULT = ROOT / "evidence/06c-gpu-localization/raw-key-replay-1/results.json"
PRIOR_RESULT = ROOT / "evidence/06d-gpu-rounding/operator-1/results.json"
CAPTURE_NAMES = ("x_9", "x_27", "x_29", "x_31")
POLICIES = ("CPU_ONLY", "CPU_AND_GPU")
LENGTHS = (1, 3, 12, 32)
WIDTH = 256


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def save(path: Path, value: dict) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")
    temp.replace(path)


def inventory(root: Path) -> dict[str, dict[str, object]]:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlink in generated model: {path}")
        if path.is_file():
            result[str(path.relative_to(root))] = {"size": path.stat().st_size,
                                                   "sha256": sha(path)}
        elif not path.is_dir():
            raise ValueError(f"Nonregular generated model entry: {path}")
    return result


def directed_vector():
    import numpy as np

    values = [np.float32(-0.0), np.float32(0.0),
              np.nextafter(np.float32(0), np.float32(1)),
              np.nextafter(np.float32(0), np.float32(-1)),
              np.float32(1e-30), np.float32(-1e-30),
              np.nextafter(np.float32(1e-30), np.float32(0)),
              np.nextafter(np.float32(-1e-30), np.float32(0)),
              np.float32(1e-40), np.float32(-1e-40)]
    # Below the clamp, Q's step is 2**-107. These six values straddle its
    # half-step and exercise ties to signed zero at the smallest active scale.
    clamp_half_step = np.float32(2.0 ** -108)
    for sign in (1, -1):
        midpoint = np.float32(sign * clamp_half_step)
        values.extend((np.nextafter(midpoint, np.float32(-np.inf)), midpoint,
                       np.nextafter(midpoint, np.float32(np.inf))))
    for exponent in (-24, -18, -10, -1, 0, 5, 8):
        boundary = np.float32(2.0 ** exponent)
        for sign in (1, -1):
            value = np.float32(sign * boundary)
            values.extend((np.nextafter(value, np.float32(-np.inf)), value,
                           np.nextafter(value, np.float32(np.inf))))
        step = np.float32(2.0 ** (exponent - 7))
        for lower in (128, 129, 254, 255):
            midpoint = np.float32((lower + 0.5) * step)
            for sign in (1, -1):
                value = np.float32(sign * midpoint)
                values.extend((np.nextafter(value, np.float32(-np.inf)), value,
                               np.nextafter(value, np.float32(np.inf))))
    # The 256-wide operator input can hold every directed point in one row.
    assert len(values) <= WIDTH
    rng = np.random.default_rng(706)
    while len(values) < WIDTH:
        exponent = int(rng.integers(-24, 9))
        mantissa = float(rng.uniform(1.0, 2.0))
        sign = -1.0 if bool(rng.integers(0, 2)) else 1.0
        values.append(np.float32(sign * mantissa * 2.0 ** exponent))
    vector = np.asarray(values, dtype=np.float32)
    assert vector.shape == (WIDTH,) and np.isfinite(vector).all()
    return vector


def cases(source: dict):
    import numpy as np

    vector = directed_vector()
    for count in LENGTHS:
        rows = [np.roll(vector, 17 * row) for row in range(count)]
        yield {"name": f"directed-q{count}", "origin": "directed finite ties, neighbors, boundaries, signed zeros and fixed-seed model range",
               "x": np.stack(rows, axis=0)[None, :, :].copy()}
    for chunk, count in ((0, 32), (1, 12)):
        for policy in ("CPU", "GPU"):
            label = f"chunk{chunk}/{policy}-tapped/outputs"
            info = source["files"][label]
            path = ROOT / info["path"]
            if sha(path) != info["sha256"]:
                raise ValueError(f"Captured array archive changed: {label}")
            with np.load(path) as stored:
                assert set(CAPTURE_NAMES) <= set(stored.files)
                for name in CAPTURE_NAMES:
                    original = stored[name]
                    assert original.dtype == np.float32 and original.shape in \
                        ((1, count, WIDTH), (1, count, 2, 128))
                    value = original.reshape(1, count, WIDTH).copy()
                    assert np.isfinite(value).all()
                    yield {"name": f"captured-c{chunk}-{policy.lower()}-{name}",
                           "origin": {"archive": label, "archive_sha256": info["sha256"],
                                      "value_name": name}, "x": value}


def compare(actual, expected) -> dict:
    import numpy as np

    same_type = actual.dtype == expected.dtype and actual.shape == expected.shape
    if not same_type:
        return {"exact": False, "shape": list(actual.shape), "dtype": str(actual.dtype),
                "expected_shape": list(expected.shape), "expected_dtype": str(expected.dtype),
                "different_values": None, "examples": []}
    a = np.ascontiguousarray(actual).view(np.uint32)
    b = np.ascontiguousarray(expected).view(np.uint32)
    indexes = np.argwhere(a != b)
    examples = []
    for row in indexes[:8]:
        index = tuple(int(i) for i in row)
        examples.append({"index": list(index), "actual": float(actual[index]),
                         "expected": float(expected[index]),
                         "actual_bits": f"0x{int(a[index]):08x}",
                         "expected_bits": f"0x{int(b[index]):08x}"})
    return {"exact": not len(indexes), "shape": list(actual.shape),
            "dtype": str(actual.dtype), "different_values": int(len(indexes)),
            "examples": examples}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args()
    if args.attempt != "operator-2":
        parser.error("This bounded protocol permits only fresh --attempt operator-2")
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = read(protocol_path)
    helper = ROOT / "scripts/signed_even_round_v2.py"
    reference_source = ROOT / "coreml_port/graphs.py"
    assert protocol["schema_version"] == 1 and protocol["attempt"] == "operator-2"
    assert protocol["script_sha256"] == sha(Path(__file__))
    assert protocol["helper_sha256"] == sha(helper)
    assert protocol["reference_source_sha256"] == sha(reference_source)
    assert protocol["source_result_sha256"] == sha(SOURCE_RESULT)
    assert protocol["prior_result_sha256"] == sha(PRIOR_RESULT)
    assert protocol["query_lengths"] == list(LENGTHS) and protocol["width"] == WIDTH
    assert protocol["capture_names"] == list(CAPTURE_NAMES)
    evidence = ROOT / "evidence/06d-gpu-rounding" / args.attempt
    artifacts = ROOT / "artifacts/06d-gpu-rounding" / args.attempt
    if evidence.exists() or artifacts.exists():
        raise FileExistsError("Fresh operator evidence/artifact attempt required")
    evidence.mkdir(parents=True, exist_ok=False)
    artifacts.mkdir(parents=True, exist_ok=False)
    (evidence / "source-snapshot.py").write_bytes(Path(__file__).read_bytes())
    (evidence / "helper-snapshot.py").write_bytes(helper.read_bytes())
    (evidence / "protocol.json").write_bytes(protocol_path.read_bytes())
    report = {"status": "starting", "passed": False, "script_sha256": sha(Path(__file__)),
              "helper_sha256": sha(helper), "protocol_sha256": sha(protocol_path),
              "reference_source_sha256": sha(reference_source),
              "source_result_sha256": sha(SOURCE_RESULT),
              "prior_result_sha256": sha(PRIOR_RESULT),
              "finite_input_contract": "All exercised arrays are finite FP32; NaN and infinities excluded because the indexed-step baseline is not defined for them.",
              "actual_device_placement": "unverified; compute policy only", "cases": [],
              "native_predictions": 0}

    def update() -> None:
        save(evidence / "results.json", report)

    update()
    models = {}
    try:
        import coremltools as ct
        import numpy as np
        import torch
        from torch import nn
        from coreml_port.graphs import Q
        from scripts.signed_even_round_v2 import Q_signed_even_v2
        from coreml_port.prediction_guard import install, stats
        from coreml_port.cleanup_guard import install as install_cleanup, stats as cleanup_stats

        torch.set_num_threads(4)
        source = read(SOURCE_RESULT)
        prior = read(PRIOR_RESULT)
        assert source["status"] == "completed" and source["passed"] and \
            source["instrumentation_valid"] and source["previous_tap_compatible"]
        prepared = list(cases(source))
        assert len(prepared) == 4 + 2 * 2 * len(CAPTURE_NAMES)
        assert [case["name"] for case in prepared] == [case["name"] for case in prior["cases"]]
        for case, old in zip(prepared, prior["cases"]):
            assert hashlib.sha256(case["x"].tobytes()).hexdigest() == old["input_sha256"]
        torch_failures = 0
        for case in prepared:
            x = case["x"]
            assert x.dtype == np.float32 and x.shape[0] == 1 and x.shape[2] == WIDTH and \
                x.shape[1] in LENGTHS and np.isfinite(x).all()
            with torch.inference_mode():
                tensor = torch.from_numpy(x.copy())
                reference = Q(tensor).numpy().copy()
                candidate = Q_signed_even_v2(tensor).numpy().copy()
            result = compare(candidate, reference)
            torch_failures += int(not result["exact"])
            path = evidence / "cases" / f"{case['name']}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, x=x, reference=reference, torch_candidate=candidate)
            report["cases"].append({"name": case["name"], "origin": case["origin"],
                                    "count": int(x.shape[1]), "input_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
                                    "case_archive": str(path.relative_to(ROOT)),
                                    "case_archive_sha256": sha(path),
                                    "input_finite": True,
                                    "input_negative_zero_count": int(np.count_nonzero((x == 0) & np.signbit(x))),
                                    "torch_vs_reference": result,
                                    "native": {}})
            update()
        report["torch_case_failures"] = torch_failures
        report["torch_gate_passed"] = torch_failures == 0
        if torch_failures:
            raise AssertionError("Torch signed-even candidate differs from sealed Q; all cases recorded")

        class Operator(nn.Module):
            def forward(self, x):
                return Q_signed_even_v2(x)

        install_cleanup()
        install()
        assert stats() == {"installed": True, "plain_calls": 0,
                           "state_calls": 0, "gil_releases": 0}
        graph = Operator().eval()
        sample = torch.from_numpy(prepared[1]["x"].copy())
        with torch.inference_mode():
            traced = torch.jit.trace(graph, sample, check_trace=False)
        started = time.perf_counter()
        converted = ct.convert(
            traced,
            inputs=[ct.TensorType(name="x", shape=(1, ct.RangeDim(1, 32, default=3), WIDTH), dtype=np.float32)],
            outputs=[ct.TensorType(name="y", dtype=np.float32)],
            convert_to="mlprogram", compute_precision=ct.precision.FLOAT32,
            minimum_deployment_target=ct.target.macOS15, skip_model_load=True)
        package = artifacts / "signed-even-v2.mlpackage"
        converted.save(str(package))
        report["conversion_seconds"] = time.perf_counter() - started
        report["package"] = str(package.relative_to(ROOT))
        report["package_files"] = inventory(package)
        update()
        compiled = artifacts / "signed-even-v2.mlmodelc"
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
            for case, row in zip(prepared, report["cases"]):
                if sha(ROOT / row["case_archive"]) != row["case_archive_sha256"]:
                    raise ValueError(f"Prepared case archive changed: {row['name']}")
                with np.load(ROOT / row["case_archive"]) as archived:
                    x = archived["x"].copy()
                    expected = archived["reference"].copy()
                result = models[policy].predict({"x": x})
                assert set(result) == {"y"}
                actual = result["y"].copy()
                output_path = evidence / "native" / policy / f"{row['name']}.npz"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(output_path, actual=actual)
                row["native"][policy] = {"comparison": compare(actual, expected),
                                         "output_archive": str(output_path.relative_to(ROOT)),
                                         "output_sha256": sha(output_path),
                                         "finite": bool(np.isfinite(actual).all())}
                report["native_predictions"] += 1
                update()
            del models[policy]
            gc.collect()
        models.clear()
        report["prediction_guard"] = stats()
        report["cleanup_guard"] = cleanup_stats()
        assert stats() == {"installed": True, "plain_calls": len(prepared) * len(POLICIES),
                           "state_calls": 0, "gil_releases": len(prepared) * len(POLICIES)}
        report["native_case_failures"] = sum(
            not row["native"][policy]["comparison"]["exact"] or not row["native"][policy]["finite"]
            for row in report["cases"] for policy in POLICIES)
        assert inventory(package) == report["package_files"] and \
            inventory(compiled) == report["compiled_files"]
        assert sha(Path(__file__)) == report["script_sha256"] and sha(helper) == report["helper_sha256"] and \
            sha(protocol_path) == report["protocol_sha256"] and \
            sha(reference_source) == report["reference_source_sha256"] and \
            sha(SOURCE_RESULT) == report["source_result_sha256"] and \
            sha(PRIOR_RESULT) == report["prior_result_sha256"]
        report.update(status="completed", passed=report["native_case_failures"] == 0,
                      operator_gate_passed=report["native_case_failures"] == 0,
                      scope="Isolated finite FP32 quantizer; CPU_AND_GPU policy does not prove GPU execution or composed-graph fidelity")
    except BaseException as exc:
        report.update(status="failed", passed=False,
                      error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        models.clear()
        gc.collect()
        update()
    print(json.dumps({"passed": report["passed"], "cases": len(report["cases"]),
                      "native_predictions": report["native_predictions"],
                      "native_case_failures": report["native_case_failures"]}, indent=2))


if __name__ == "__main__":
    main()
