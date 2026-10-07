"""Offline FP32 reconstruction of saved layer-2 key normalization taps.

No model is loaded or run. Signed zero is compared by FP32 bit pattern.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "evidence/06g-gpu-residual/key-counterfactual-protocol.json"
DEFAULT_OUTPUT = ROOT / "evidence/06g-gpu-residual/key-counterfactual.json"
STAGES = ("x_207", "var_14339", "var_14340", "k_49")


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def record(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def bits(value: np.ndarray) -> np.ndarray:
    assert value.dtype == np.float32
    return np.ascontiguousarray(value).view(np.uint32)


def comparison(actual: np.ndarray, expected: np.ndarray) -> dict:
    assert actual.dtype == expected.dtype == np.float32 and actual.shape == expected.shape
    mismatch = bits(actual) != bits(expected)
    coords = np.argwhere(mismatch)
    first = tuple(int(x) for x in coords[0]) if len(coords) else None
    return {"byte_exact": first is None, "mismatch_count": int(mismatch.sum()),
            "first_index": list(first) if first is not None else None,
            "first_actual": float(actual[first]) if first is not None else None,
            "first_expected": float(expected[first]) if first is not None else None,
            "first_actual_bits": int(bits(actual)[first]) if first is not None else None,
            "first_expected_bits": int(bits(expected)[first]) if first is not None else None}


def fp32_path(projection: np.ndarray, seed: np.ndarray, step: np.ndarray) -> dict[str, np.ndarray]:
    assert projection.dtype == seed.dtype == step.dtype == np.float32
    assert projection.ndim == 4 and seed.shape == projection.shape[:-1] + (1,)
    assert step.shape == projection.shape and np.all(np.isfinite(projection))
    assert np.all(np.isfinite(seed)) and np.all(np.isfinite(step)) and np.all(step > 0)
    with np.errstate(all="raise"):
        raw = np.multiply(projection, seed, dtype=np.float32)
        quotient = np.divide(raw, step, dtype=np.float32)
        rounded = np.rint(quotient).astype(np.float32)
        key = np.multiply(rounded, step, dtype=np.float32)
    return dict(zip(STAGES, (raw, quotient, rounded, key), strict=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    protocol = record(args.protocol)
    assert protocol["status"] == "frozen_before_run"
    script_hash = sha(Path(__file__))
    assert script_hash == protocol["script_sha256"]
    for name, expected in protocol["pins"].items():
        assert sha(ROOT / name) == expected, name
    replay = record(ROOT / protocol["replay"])
    assert replay["status"] == "completed" and replay["passed"] is True
    assert replay["instrumentation_valid"] is True and len(replay["calls"]) == 8
    assert len(replay["controls"]) == 4 and all(c["passed"] for c in replay["controls"])
    assert replay["pins"][protocol["build"]] == protocol["pins"][protocol["build"]]
    for name, expected in replay["pins"].items():
        assert sha(ROOT / name) == expected, name
    for entry in replay["files"].values():
        assert sha(ROOT / entry["path"]) == entry["sha256"]
    calls = {(c["chunk"], c["policy"], c["arm"]): c for c in replay["calls"]}
    assert set(calls) == {(chunk, policy, arm) for chunk in (0, 1)
                          for policy in ("CPU", "GPU") for arm in ("original", "tapped")}
    control_checks = []
    outputs = {}
    for chunk in (0, 1):
        for policy in ("CPU", "GPU"):
            original = calls[(chunk, policy, "original")]
            tapped = calls[(chunk, policy, "tapped")]
            for kind, keys in (("outputs", ("y", "tap23_out")),
                               ("state_after", tuple(f"inner_{part}_cache_{layer}"
                                                     for layer in range(4) for part in ("k", "v")))):
                with np.load(ROOT / original[kind]["path"], allow_pickle=False) as a, \
                        np.load(ROOT / tapped[kind]["path"], allow_pickle=False) as b:
                    for key in keys:
                        check = comparison(a[key], b[key])
                        assert check["byte_exact"], (chunk, policy, kind, key)
                        control_checks.append({"chunk": chunk, "policy": policy,
                                               "kind": kind, "name": key, "byte_exact": True})
            with np.load(ROOT / tapped["outputs"]["path"], allow_pickle=False) as archive:
                outputs[(chunk, policy)] = {name: archive[name].copy() for name in
                                            ("value_41", "denom_21", "var_14324", "step_149", *STAGES)}
    diagonal = {}
    substitutions = {}
    for chunk in (0, 1):
        diagonal[str(chunk)] = {}
        for policy in ("CPU", "GPU"):
            arrays = outputs[(chunk, policy)]
            stages = fp32_path(arrays["value_41"], arrays["var_14324"], arrays["step_149"])
            diagonal[str(chunk)][policy] = {stage: comparison(stages[stage], arrays[stage])
                                            for stage in STAGES}
            assert all(item["byte_exact"] for item in diagonal[str(chunk)][policy].values())
        cpu, gpu = outputs[(chunk, "CPU")], outputs[(chunk, "GPU")]
        inputs = {name: comparison(gpu[name], cpu[name]) for name in
                  ("value_41", "denom_21", "step_149", "var_14324")}
        substituted = fp32_path(gpu["value_41"], cpu["var_14324"], gpu["step_149"])
        substitutions[str(chunk)] = {"inputs_gpu_vs_cpu": inputs,
                                     "counterfactual_vs_cpu": {
                                         stage: comparison(substituted[stage], cpu[stage])
                                         for stage in STAGES},
                                     "observed_gpu_vs_cpu": {
                                         stage: comparison(gpu[stage], cpu[stage])
                                         for stage in STAGES}}
    warm = substitutions["1"]
    assert all(warm["inputs_gpu_vs_cpu"][name]["byte_exact"]
               for name in ("value_41", "denom_21", "step_149"))
    assert not warm["inputs_gpu_vs_cpu"]["var_14324"]["byte_exact"]
    assert all(warm["counterfactual_vs_cpu"][stage]["byte_exact"] for stage in STAGES)
    assert not warm["observed_gpu_vs_cpu"]["k_49"]["byte_exact"]
    result = {"status": "completed", "passed": True,
              "scope": "Offline FP32 counterfactual at saved group-0 layer-2 key normalization only; no native correction or full-model claim",
              "script_sha256": script_hash,
              "protocol_sha256": sha(args.protocol),
              "replay_sha256": protocol["pins"][protocol["replay"]],
              "all_replay_files_and_pins_verified": True,
              "independent_original_tapped_endpoint_state_controls": len(control_checks),
              "controls": control_checks,
              "diagonal_saved_stage_reconstruction": diagonal,
              "substitution": substitutions,
              "warm_local_key_norm_counterfactual_passed": True,
              "interpretation": "On the saved warm tensors, substituting only the CPU reciprocal seed into the GPU FP32 key-normalization path reproduces CPU raw, quotient, rounded, and quantized key arrays byte-for-byte. This does not establish that a compiled replacement would preserve execution or fix the full model.",
              "next_design": "For denominators near 24–59, a separate power-of-two scaling experiment could map d to d/8 within the existing helper range and scale the reciprocal back by 1/8. That is a hypothesis requiring host/native exponent, seed, endpoint, and state gates; the frozen helper is unchanged."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"passed": result["passed"],
                      "warm_seed_differences": warm["inputs_gpu_vs_cpu"]["var_14324"]["mismatch_count"],
                      "warm_key_observed_differences": warm["observed_gpu_vs_cpu"]["k_49"]["mismatch_count"],
                      "warm_key_after_substitution_differences": warm["counterfactual_vs_cpu"]["k_49"]["mismatch_count"]}))


if __name__ == "__main__":
    main()
