"""Validate and summarize four fresh-process compiled-cache startup probes.

This reads saved probe evidence only. OS/filesystem caches were not controlled;
the comparison is descriptive and says nothing about decode throughput.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE = ROOT / "evidence/05c-compiled"
RUNS = ("compiled-first", "package-control", "compiled-reopen1", "compiled-reopen2")
COMPILED_RUNS = ("compiled-first", "compiled-reopen1", "compiled-reopen2")
COMPONENTS = ("embedding", *(str(index) for index in range(12)), "head")
TIMINGS = ("entry_to_imports_seconds", "initialization_seconds",
           "entry_to_ready_seconds", "entry_to_first_scores_seconds",
           "request_create_seconds", "prefill32_seconds", "decode_seconds",
           "entry_to_finish_seconds")
COMPARISON = ("entry_to_ready_seconds", "entry_to_first_scores_seconds",
              "initialization_seconds", "component_constructor_sum_seconds",
              "peak_process_rss_bytes")


class SummaryError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SummaryError(message)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def seconds(value, label: str) -> float:
    require(type(value) in (float, int) and math.isfinite(value) and value >= 0,
            f"Invalid nonnegative timing: {label}")
    return float(value)


def digest(value, label: str) -> str:
    require(isinstance(value, str) and len(value) == 64 and
            all(c in "0123456789abcdef" for c in value), f"Invalid SHA-256: {label}")
    return value


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SummaryError(f"Cannot read {path}: {exc}") from exc
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def summarize(evidence: Path) -> dict:
    """Return an evidence-checked summary without writing or loading models."""
    records = {name: load_json(evidence / name / "results.json") for name in RUNS}
    seal = load_json(evidence / "full-cache-1/compiled-identity.json")
    pinned_sources = load_json(evidence / "cohort-source-identity.json")
    candidate = digest(seal.get("source_candidate_identity"), "sealed source candidate")
    compiled_aggregate = digest(seal.get("aggregate_sha256"), "sealed compiled cache")
    probe_sha = digest(pinned_sources.get("scripts/probe_compiled_pipeline.py"),
                       "pinned probe source")
    manifest_sha = sha256(evidence / "full-cache-1/compiled-identity.json")
    base = records["compiled-first"]
    require(isinstance(base.get("runtime_provenance"), dict) and
            bool(base["runtime_provenance"]), "Missing original runtime provenance")
    for source, value in base["runtime_provenance"].items():
        digest(value, f"runtime source {source}")

    fixture_fields = ("input_fixture_sha256", "score_fixture_sha256", "prompt_ids",
                      "forced_next_token")
    require(isinstance(base.get("prompt_ids"), list) and len(base["prompt_ids"]) == 32
            and all(type(token) is int and token >= 0 for token in base["prompt_ids"]),
            "Invalid 32-token input fixture")
    require(type(base.get("forced_next_token")) is int and
            base["forced_next_token"] >= 0, "Invalid forced next token")
    for field in fixture_fields[:2]:
        digest(base.get(field), field)
    require(base["input_fixture_sha256"] == sha256(ROOT / "evidence/04-coreml/generation-comparison.json"),
            "Current input fixture differs from probe fixture")
    require(base["score_fixture_sha256"] == sha256(ROOT / "evidence/05-practical/coreml-pilot-1/scores.npz"),
            "Current score fixture differs from probe fixture")

    original_guard = None
    compiled_loader = None
    result_runs = {}
    for name, row in records.items():
        require(row.get("status") == "completed" and row.get("passed") is True and
                row.get("first_scores_exact") is True and
                row.get("decode_scores_exact") is True and
                row.get("request_disposed") is True,
                f"Incomplete, inexact, or undisposed probe: {name}")
        require(row.get("script_sha256") == probe_sha, f"Probe source mismatch: {name}")
        snapshot = evidence / name / "source-snapshot.py"
        require(sha256(snapshot) == probe_sha, f"Probe snapshot mismatch: {name}")
        require(row.get("runtime_provenance") == base["runtime_provenance"],
                f"Original runtime provenance mismatch: {name}")
        identity = row.get("package_identity")
        require(isinstance(identity, dict) and
                identity.get("aggregate_sha256") == candidate,
                f"Source candidate identity mismatch: {name}")
        for field in fixture_fields:
            require(row.get(field) == base[field], f"Input/score fixture mismatch: {name}:{field}")
        guard = row.get("prediction_guard")
        require(isinstance(guard, dict) and guard.get("installed") is True,
                f"Prediction guard absent: {name}")
        for field in ("plain_calls", "state_calls", "gil_releases"):
            require(type(guard.get(field)) is int and guard[field] > 0,
                    f"Missing guarded call count: {name}:{field}")
        require(guard["gil_releases"] >= guard["plain_calls"] + guard["state_calls"],
                f"Guard GIL release count too small: {name}")
        call_counts = {field: guard[field] for field in
                       ("plain_calls", "state_calls", "gil_releases")}
        if original_guard is None:
            original_guard = call_counts
        require(call_counts == original_guard, f"Guarded call counts differ: {name}")

        loader = row.get("loader_provenance")
        require(isinstance(loader, dict), f"Missing loader provenance: {name}")
        if name in COMPILED_RUNS:
            require(row.get("runtime_adapter") == "compiled_tiny_prefix_replay" and
                    loader.get("mode") == "persistent_compiled", f"Wrong compiled mode: {name}")
            require(loader.get("source_candidate_identity") == candidate and
                    loader.get("compiled_aggregate_sha256") == compiled_aggregate and
                    loader.get("compiled_manifest_sha256") == manifest_sha,
                    f"Compiled loader identity mismatch: {name}")
            sources = loader.get("sources")
            require(isinstance(sources, dict) and bool(sources),
                    f"Missing compiled loader sources: {name}")
            for source, value in sources.items():
                digest(value, f"compiled loader source {source}")
                if source in pinned_sources:
                    require(value == pinned_sources[source], f"Unpinned compiled loader source: {source}")
            if compiled_loader is None:
                compiled_loader = loader
            require(loader == compiled_loader, f"Compiled loader provenance differs: {name}")
        else:
            require(row.get("runtime_adapter") == "tiny_prefix_replay" and
                    loader.get("mode") == "mlpackage", "Package control used wrong loader")

        timings = {field: seconds(row.get(field), f"{name}:{field}") for field in TIMINGS}
        require(timings["entry_to_imports_seconds"] <= timings["entry_to_ready_seconds"] <=
                timings["entry_to_first_scores_seconds"] <= timings["entry_to_finish_seconds"],
                f"Invalid startup milestone order: {name}")
        require(timings["initialization_seconds"] <= timings["entry_to_ready_seconds"],
                f"Initialization exceeds entry-to-ready: {name}")
        components = row.get("component_load_seconds")
        require(isinstance(components, dict) and set(components) == set(COMPONENTS),
                f"Missing or extra component constructors: {name}")
        component_sum = sum(seconds(value, f"{name}:component:{component}")
                            for component, value in components.items())
        require(component_sum <= timings["initialization_seconds"] + 1e-6,
                f"Component constructors exceed initialization: {name}")
        identity_seconds = row.get("identity_verification_seconds")
        source_spec_seconds = row.get("source_spec_seconds")
        if name in COMPILED_RUNS:
            identity_seconds = seconds(identity_seconds, f"{name}:identity")
            source_spec_seconds = seconds(source_spec_seconds, f"{name}:source-spec")
        else:
            require(identity_seconds is None and source_spec_seconds is None,
                    "Package control unexpectedly reports compiled-only timing fields")
        attributed = component_sum + (identity_seconds or 0) + (source_spec_seconds or 0)
        require(attributed <= timings["initialization_seconds"] + 1e-6,
                f"Constructor/identity timings exceed initialization: {name}")
        rss = row.get("peak_process_rss_bytes")
        require(type(rss) is int and rss > 0, f"Invalid peak process RSS: {name}")
        result_runs[name] = {
            "status": row["status"], "mode": loader["mode"],
            "host": row.get("host"), "cache_state": row.get("cache_state"),
            "timings_seconds": {**timings,
                "component_constructor_sum_seconds": component_sum,
                "identity_verification_seconds": identity_seconds,
                "source_spec_seconds": source_spec_seconds,
                "initialization_unattributed_seconds": timings["initialization_seconds"] - attributed},
            "component_load_seconds": components,
            "peak_process_rss_bytes": rss,
            "prediction_guard": guard,
            "score_checks": {"first_scores_exact": True, "decode_scores_exact": True,
                             "request_disposed": True},
        }

    control = result_runs["package-control"]
    reopens = [result_runs[name] for name in ("compiled-reopen1", "compiled-reopen2")]
    comparison = {}
    for field in COMPARISON:
        def measured(run: dict) -> float:
            return (run["peak_process_rss_bytes"] if field == "peak_process_rss_bytes"
                    else run["timings_seconds"][field])
        control_value = measured(control)
        reopen_values = [measured(run) for run in reopens]
        reopen_median = median(reopen_values)
        comparison[field] = {"package_control": control_value,
                             "compiled_reopens": reopen_values,
                             "compiled_reopen_median": reopen_median,
                             "control_minus_reopen_median": control_value - reopen_median,
                             "reopen_median_over_control":
                                 reopen_median / control_value if control_value else None}

    return {
        "status": "pass", "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_candidate_identity": candidate,
        "compiled_aggregate_sha256": compiled_aggregate,
        "compiled_manifest_sha256": manifest_sha,
        "probe_script_sha256": probe_sha,
        "runtime_provenance": base["runtime_provenance"],
        "compiled_loader_provenance": compiled_loader,
        "common_fixture": {field: base[field] for field in fixture_fields},
        "runs": result_runs,
        "comparison": {
            "design": "one package control versus median of two later compiled fresh-process reopens",
            "first_compiled_open": "compiled-first is reported separately and excluded from reopen median",
            "metrics": comparison,
        },
        "limits": [
            "OS/filesystem caches and concurrent host load were not controlled.",
            "Sample sizes are one package control and two compiled reopens; these are descriptive observations.",
            "Compiled and package loaders intentionally incur different identity verification overhead.",
            "The probes do not support a decode speedup claim.",
            "RSS is process ru_maxrss high-water mark, not a steady-state or system-wide measure.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    evidence = args.evidence_root.expanduser().resolve()
    output = args.output or evidence / "startup-summary.json"
    if output.exists():
        parser.error(f"Output exists; refusing to overwrite: {output}")
    result = summarize(evidence)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n",
                      encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(output),
                      "source_candidate_identity": result["source_candidate_identity"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
