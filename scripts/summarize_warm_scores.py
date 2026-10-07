"""Validate and summarize the bounded 05f warm-score benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PREFIXES = (32, 128, 512)
EXPECTED_ARMS = ("A", "B")
EXPECTED_OUTPUTS = {"y", "tap23_out"}
EXPECTED_STATES = {f"inner_{kv}_cache_{layer}" for layer in range(4) for kv in ("k", "v")}
WARM_ORDER = "ABBAABBA"
TIMING_ORDER = "ABBA"
BLOCK_PREFIX_ORDER = {0: (32, 128, 512), 1: (512, 32, 128), 2: (128, 512, 32)}
THRESHOLD = 0.10
TIMING_BOUNDARY = "Synchronous CompiledMLModel.predict including bridge/output conversion; fresh state creation/restoration and exact checks excluded"
EXPECTED_GUARD = {"installed": True, "plain_calls": 0, "state_calls": 180, "gil_releases": 180}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _exact_map(value: Any, expected_keys: set[str]) -> bool:
    return isinstance(value, dict) and set(value) == expected_keys and all(v is True for v in value.values())


def _finite_positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def _close_recorded(actual: Any, expected: float) -> bool:
    return isinstance(actual, (int, float)) and not isinstance(actual, bool) and math.isclose(
        float(actual), expected, rel_tol=1e-12, abs_tol=1e-12,
    )


def _validate_protocol(protocol: dict[str, Any], errors: list[str]) -> None:
    if protocol.get("go_median_reduction") != THRESHOLD:
        errors.append("protocol threshold must remain the predeclared 10% median reduction")
    if protocol.get("cycles_per_block") != 4 or protocol.get("blocks") != 3:
        errors.append("protocol must specify 4 ABBA cycles in each of 3 blocks")
    fixtures = protocol.get("fixtures")
    if not isinstance(fixtures, list) or len(fixtures) != 3 or any(
        not isinstance(item, dict) or not isinstance(item.get("results"), str) or
        not isinstance(item.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
        for item in (fixtures or [])
    ):
        errors.append("protocol must bind exactly three fixture result files by SHA-256")
    elif all("prefix" in item for item in fixtures):
        if [item.get("prefix") for item in fixtures] != list(EXPECTED_PREFIXES):
            errors.append("protocol fixture set must be ordered prefixes 32, 128, 512")
    scope = protocol.get("scope")
    if not isinstance(scope, str) or "group0" not in scope or not all(str(p) in scope for p in EXPECTED_PREFIXES):
        errors.append("protocol scope must identify group0 positions 32, 128, and 512")
    if not isinstance(protocol.get("script_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", protocol["script_sha256"]):
        errors.append("protocol benchmark script SHA-256 is missing or malformed")
    if not isinstance(protocol.get("build_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", protocol["build_sha256"]):
        errors.append("protocol variant build SHA-256 is missing or malformed")
    if not isinstance(protocol.get("build"), str) or not protocol["build"]:
        errors.append("protocol variant build record path is missing")


def summarize_result(
    result: dict[str, Any],
    protocol: dict[str, Any],
    *,
    current_script_sha256: str | None = None,
    protocol_file_sha256: str | None = None,
) -> dict[str, Any]:
    """Return a validation result; a valid below-threshold run remains completed."""
    errors: list[str] = []
    _validate_protocol(protocol, errors)
    if result.get("protocol") != protocol:
        errors.append("recorded protocol differs from the selected frozen protocol")
    if result.get("protocol_sha256") != protocol_file_sha256 and protocol_file_sha256 is not None:
        errors.append("recorded protocol file digest differs from the selected protocol")
    script_sha = result.get("script_sha256")
    if script_sha != protocol.get("script_sha256"):
        errors.append("run script digest differs from the predeclared script digest")
    if current_script_sha256 is not None and script_sha != current_script_sha256:
        errors.append("run script digest differs from the current benchmark script")
    if result.get("status") != "completed" or result.get("experiment_completed") is not True:
        errors.append("benchmark did not record a completed experiment")
    if result.get("timing_boundary") != TIMING_BOUNDARY:
        errors.append("timing boundary differs from the predeclared synchronous prediction interval")
    if result.get("baseline_preservation") is not True:
        errors.append("baseline preservation check is missing or failed")
    baseline_identity = result.get("baseline_identity")
    if not isinstance(baseline_identity, str) or not re.fullmatch(r"[0-9a-f]{64}", baseline_identity):
        errors.append("baseline compiled identity is missing or malformed")
    elif protocol.get("baseline_identity") is not None and baseline_identity != protocol["baseline_identity"]:
        errors.append("baseline identity differs from the predeclared protocol")
    if result.get("scope_rejections") != 9:
        errors.append("expected all 9 out-of-scope input rejections")
    if result.get("exactness_passed") is not True:
        errors.append("pre-timing exactness flag is missing or false")
    runtime_provenance = result.get("runtime_provenance")
    if not isinstance(runtime_provenance, dict) or not runtime_provenance:
        errors.append("runtime/source provenance is missing")
    expected_runtime = protocol.get("runtime_provenance")
    if expected_runtime is not None and runtime_provenance != expected_runtime:
        errors.append("runtime/source provenance differs from the predeclared protocol")
    if result.get("prediction_guard") != EXPECTED_GUARD:
        errors.append("prediction guard must record exactly 180 guarded state calls and GIL releases")

    correctness = result.get("correctness")
    expected_correctness = [(prefix, repeat, arm)
                            for prefix in EXPECTED_PREFIXES for repeat in range(2) for arm in EXPECTED_ARMS]
    observed_correctness = []
    if not isinstance(correctness, list) or len(correctness) != 12:
        errors.append("expected exactly 12 initial exactness calls")
        correctness = correctness if isinstance(correctness, list) else []
    for row in correctness:
        if not isinstance(row, dict):
            errors.append("correctness entry is not an object")
            continue
        observed_correctness.append((row.get("prefix"), row.get("repeat"), row.get("arm")))
        if row.get("passed") is not True or not _exact_map(row.get("outputs_exact"), EXPECTED_OUTPUTS) or \
                not _exact_map(row.get("states_exact"), EXPECTED_STATES):
            errors.append(f"initial exactness call failed or omitted exact output/state maps: {row.get('prefix')}/{row.get('repeat')}/{row.get('arm')}")
        if not _finite_positive(row.get("seconds")):
            errors.append("initial exactness call duration must be finite and positive")
    if observed_correctness != expected_correctness:
        errors.append("initial exactness calls are missing, duplicated, or out of prefix/repeat/arm order")

    warmup = result.get("warmup")
    expected_warmup = [(prefix, arm) for prefix in EXPECTED_PREFIXES for arm in WARM_ORDER]
    observed_warmup = []
    if not isinstance(warmup, list) or len(warmup) != 24:
        errors.append("expected exactly 24 warmup calls")
        warmup = warmup if isinstance(warmup, list) else []
    for row in warmup:
        if not isinstance(row, dict):
            errors.append("warmup entry is not an object")
            continue
        observed_warmup.append((row.get("prefix"), row.get("arm")))
        if not row.get("passed") is True or not _exact_map(row.get("outputs_exact"), EXPECTED_OUTPUTS) or \
                not _exact_map(row.get("states_exact"), EXPECTED_STATES):
            errors.append(f"warmup replay failed exactness: {row.get('prefix')}/{row.get('arm')}")
        if not _finite_positive(row.get("seconds")):
            errors.append("warmup duration must be finite and positive")
    if observed_warmup != expected_warmup:
        errors.append("warmup calls are missing, duplicated, or not ordered ABBAABBA by prefix")

    timings = result.get("timings")
    expected_timing_keys = [(block, prefix, cycle, slot, arm)
                            for block, prefixes in BLOCK_PREFIX_ORDER.items()
                            for prefix in prefixes
                            for cycle in range(4)
                            for slot, arm in enumerate(TIMING_ORDER)]
    observed_timing_keys = []
    if not isinstance(timings, list) or len(timings) != 144:
        errors.append("expected exactly 144 timed calls")
        timings = timings if isinstance(timings, list) else []
    for row in timings:
        if not isinstance(row, dict):
            errors.append("timing entry is not an object")
            continue
        observed_timing_keys.append((row.get("block"), row.get("prefix"), row.get("cycle"), row.get("slot"), row.get("arm")))
        if row.get("passed") is not True or not _exact_map(row.get("outputs_exact"), EXPECTED_OUTPUTS) or \
                not _exact_map(row.get("states_exact"), EXPECTED_STATES):
            errors.append(f"timed replay failed exactness: {row.get('block')}/{row.get('prefix')}/{row.get('cycle')}/{row.get('slot')}")
        if not _finite_positive(row.get("seconds")):
            errors.append("timed duration must be finite and positive")
    if observed_timing_keys != expected_timing_keys:
        errors.append("timed calls are missing, duplicated, or violate the declared block/prefix/ABBA order")

    computed_blocks = []
    for block in range(3):
        for prefix in EXPECTED_PREFIXES:
            rows = [row for row in timings if isinstance(row, dict) and row.get("block") == block and row.get("prefix") == prefix]
            arm_seconds = {arm: [row.get("seconds") for row in rows if row.get("arm") == arm] for arm in EXPECTED_ARMS}
            if any(len(values) != 8 for values in arm_seconds.values()):
                errors.append(f"block {block} prefix {prefix} must contain 8 calls per arm")
                continue
            if not all(_finite_positive(value) for values in arm_seconds.values() for value in values):
                errors.append(f"block {block} prefix {prefix} contains an invalid duration")
                continue
            base_median = median(arm_seconds["A"])
            variant_median = median(arm_seconds["B"])
            reduction = 1.0 - variant_median / base_median
            computed_blocks.append({"block": block, "prefix": prefix,
                                    "baseline_median_seconds": base_median,
                                    "variant_median_seconds": variant_median,
                                    "median_reduction": reduction,
                                    "go": reduction >= THRESHOLD})

    recorded_blocks = result.get("blocks")
    if not isinstance(recorded_blocks, list) or len(recorded_blocks) != 9:
        errors.append("expected exactly 9 recorded block-by-prefix decisions")
        recorded_blocks = recorded_blocks if isinstance(recorded_blocks, list) else []
    recorded_keys = [(row.get("block"), row.get("prefix")) for row in recorded_blocks if isinstance(row, dict)]
    expected_block_keys = [(row["block"], row["prefix"]) for row in computed_blocks]
    if recorded_keys != expected_block_keys:
        errors.append("recorded block decisions are missing, duplicated, or out of order")
    for observed, computed in zip(recorded_blocks, computed_blocks):
        if not isinstance(observed, dict):
            errors.append("recorded block decision is not an object")
            continue
        for key in ("baseline_median_seconds", "variant_median_seconds", "median_reduction"):
            if not _close_recorded(observed.get(key), computed[key]):
                errors.append(f"recorded {key} differs from recomputation for block {computed['block']} prefix {computed['prefix']}")
        if observed.get("go") is not computed["go"]:
            errors.append(f"recorded go decision differs from recomputed threshold for block {computed['block']} prefix {computed['prefix']}")
    performance_gate = len(computed_blocks) == 9 and all(row["go"] for row in computed_blocks)
    if result.get("performance_gate_passed") is not performance_gate:
        errors.append("overall performance gate flag differs from the recomputed all-nine decision")
    expected_decision = "local performance go; full integration still unqualified" if performance_gate else "no-go at predeclared threshold; do not integrate"
    if result.get("decision") != expected_decision:
        errors.append("recorded decision differs from the recomputed predeclared result")

    experiment_valid = not errors
    # An observed performance no-go is a complete outcome, not an invalid run.
    status = "completed" if experiment_valid else "invalid"
    return {
        "status": status,
        "experiment_valid": experiment_valid,
        "experiment_completed": experiment_valid,
        "performance_gate_passed": performance_gate if len(computed_blocks) == 9 else False,
        "decision": expected_decision if len(computed_blocks) == 9 else None,
        "correctness_call_count": len(correctness),
        "warmup_call_count": len(warmup),
        "timed_call_count": len(timings),
        "block_count": len(computed_blocks),
        "threshold": THRESHOLD,
        "blocks": computed_blocks,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="Completed raw benchmark results JSON.")
    parser.add_argument("--protocol", required=True, help="Exact predeclared protocol JSON used for the run.")
    parser.add_argument("--output", required=True, help="New summary JSON path; never overwritten.")
    args = parser.parse_args()
    results_path = Path(args.results).expanduser().resolve()
    protocol_path = Path(args.protocol).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite summary: {output_path}")
    if not output_path.parent.is_dir():
        raise FileNotFoundError(f"Output parent directory must already exist: {output_path.parent}")
    result = json.loads(results_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    build_errors: list[str] = []
    build_path = Path(protocol.get("build", "")).expanduser()
    if not build_path.is_absolute():
        build_path = ROOT / build_path
    try:
        build_path = build_path.resolve(strict=True)
        if sha256(build_path) != protocol.get("build_sha256"):
            build_errors.append("variant build record digest differs from the predeclared protocol")
        build = json.loads(build_path.read_text())
        if build.get("passed") is not True or build.get("graph_check", {}).get("passed") is not True:
            build_errors.append("variant build or graph check did not pass")
    except Exception as exc:
        build_errors.append(f"cannot verify variant build record: {type(exc).__name__}: {exc}")
    summary = summarize_result(
        result, protocol,
        current_script_sha256=sha256(ROOT / "scripts/benchmark_warm_scores.py"),
        protocol_file_sha256=sha256(protocol_path),
    )
    if build_errors:
        summary["errors"].extend(build_errors)
        summary["experiment_valid"] = False
        summary["experiment_completed"] = False
        summary["status"] = "invalid"
    document = {
        "schema_version": 1,
        "results_path": str(results_path),
        "results_sha256": sha256(results_path),
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "build_path": str(build_path),
        "build_sha256": sha256(build_path) if build_path.is_file() else None,
        "build_errors": build_errors,
        "summarizer_sha256": sha256(Path(__file__)),
        **summary,
    }
    output_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": document["status"], "experiment_valid": document["experiment_valid"],
                      "performance_gate_passed": document["performance_gate_passed"],
                      "decision": document["decision"], "errors": document["errors"],
                      "output": str(output_path)}, indent=2))
    return 0 if document["experiment_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
