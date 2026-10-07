"""Prepared CPU Core ML tiny/irregular partition audit against full-prefix CPU.

This script intentionally requires an explicit candidate directory and its
identity manifest. Do not run until the complete candidate passes its short
Core ML numerical gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from coreml_port.loader_selection import add_loader_arguments, create_pipeline, runtime_adapter_name, loader_provenance
from portable.model import load_model
from scripts.probe_full_model import metrics
from scripts.runtime_provenance import runtime_provenance


FROZEN_ACCEPTANCE_SHA256 = "cb4d315266126df4a98194f73f8c10671310717c06ebfd191d3dac2334bf4233"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_evidence_path(value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("--evidence-dir must be a safe repository-relative path")
    output = (ROOT / relative).resolve()
    evidence_root = (ROOT / "evidence/04d-audit").resolve()
    if not output.is_relative_to(evidence_root):
        raise ValueError("--evidence-dir must be under evidence/04d-audit")
    return output


def score_record(candidate: np.ndarray, reference: torch.Tensor, valid_vocab: int, limits: dict) -> dict:
    candidate_valid = candidate.reshape(-1)[:valid_vocab]
    reference_valid = reference.reshape(-1)[:valid_vocab]
    score_metrics = metrics(torch.from_numpy(candidate_valid.copy()), reference_valid)
    candidate_top = torch.from_numpy(candidate_valid.copy()).topk(5)
    reference_top = reference_valid.topk(5)
    margin = float(reference_top.values[0] - reference_top.values[1])
    overlap = len(set(candidate_top.indices.tolist()) & set(reference_top.indices.tolist()))
    top1_match = int(candidate_top.indices[0]) == int(reference_top.indices[0])
    ranking_pass = top1_match or margin <= 2 * limits["score_max_abs"]
    return {
        **score_metrics,
        "top1_match": top1_match,
        "reference_top1_margin": margin,
        "top5_overlap": overlap,
        "ranking_pass": ranking_pass,
        "passed": score_metrics["max_abs"] <= limits["score_max_abs"] and ranking_pass,
    }


def hidden_record(candidate: np.ndarray, reference: torch.Tensor, limits: dict) -> dict:
    value = metrics(torch.from_numpy(candidate.copy()), reference)
    value["passed"] = value["relative_rmse"] <= limits["hidden_relative_rmse"] and value["cosine"] >= limits["hidden_min_cosine"]
    return value


def record_failure(result: dict, save, stage: str, exc: BaseException) -> None:
    result["status"] = "failed"
    result["failure"] = {"stage": stage, "exception_type": type(exc).__name__, "message": str(exc)}
    result["passed"] = False
    save()


def make_cases(portable_results: dict) -> list[dict]:
    by_name = {case["name"]: case for case in portable_results["cases"]}
    tiny33 = by_name["tiny_33"]
    ids33 = tiny33["input_ids"][0]
    cases = [
        {"name": f"single_prefill_{length}", "input_ids": ids33[:length], "partitions": [length], "source": "cpu-partitions-v3/tiny_33"}
        for length in [1, 3, 15, 16, 31, 32, 33]
    ]
    for length in [1, 3, 15, 31, 33]:
        source = by_name[f"tiny_{length}"]
        cases.append({"name": f"single_token_partitions_{length}", "input_ids": source["input_ids"][0], "partitions": source["partitions"], "source": f"cpu-partitions-v3/tiny_{length}"})
    irregular = by_name["irregular_127"]
    cases.append({"name": "irregular_127", "input_ids": irregular["input_ids"][0], "partitions": irregular["partitions"], "source": "cpu-partitions-v3/irregular_127"})
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="Explicit candidate package root.")
    parser.add_argument("--identity-manifest", required=True, help="Manifest for that exact candidate package root.")
    parser.add_argument("--evidence-dir", required=True, help="New unique directory under evidence/04d-audit/.")
    parser.add_argument("--tiny-prefix-replay", action="store_true", help="Opt into CPU-equivalent cached-prefix replay below 16 tokens.")
    add_loader_arguments(parser)
    args = parser.parse_args()

    output = safe_evidence_path(args.evidence_dir)
    output.mkdir(parents=True, exist_ok=False)
    acceptance_path = ROOT / "evidence/03-portable/ACCEPTANCE.json"
    acceptance_hash = sha256(acceptance_path)
    if acceptance_hash != FROZEN_ACCEPTANCE_SHA256:
        raise RuntimeError(f"Frozen acceptance changed: {acceptance_hash}")
    calibration_path = ROOT / "evidence/03-portable/calibration-cpu.json"
    calibration = json.loads(calibration_path.read_text())
    limits = calibration["frozen_limits"]
    if not limits.get("calibration_valid"):
        raise RuntimeError("Frozen CPU reference calibration is invalid.")

    partition_path = ROOT / "evidence/03b-cache/cpu-partitions-v3.json"
    partition_source = json.loads(partition_path.read_text())
    cases = make_cases(partition_source)
    model_dir = Path(args.model_dir).expanduser()
    manifest_path = Path(args.identity_manifest).expanduser()
    result = {
        "status": "starting",
        "compute": "CPU_ONLY",
        "runtime_adapter": runtime_adapter_name(args),
        "loader_provenance": loader_provenance(args),
        "model_dir_argument": str(model_dir),
        "identity_manifest_argument": str(manifest_path),
        "thresholds_changed": False,
        "acceptance_sha256": acceptance_hash,
        "calibration_sha256": sha256(calibration_path),
        "portable_partition_fixture_sha256": sha256(partition_path),
        "validator_sha256": sha256(Path(__file__)),
        "runtime_sha256": sha256(ROOT / "coreml_port/runtime.py"),
        "tiny_prefix_runtime_sha256": sha256(ROOT / "coreml_port/tiny_prefix_runtime.py"),
        "package_identity_helper_sha256": sha256(ROOT / "coreml_port/package_identity.py"),
        "runtime_provenance": runtime_provenance(),
        "limits": limits,
        "case_plan": [{"name": c["name"], "input_ids": c["input_ids"], "partitions": c["partitions"], "fixture": c["source"]} for c in cases],
        "cases": [],
    }
    result_path = output / "results.json"

    def save() -> None:
        temporary = output / "results.json.tmp"
        temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        temporary.replace(result_path)

    save()
    start_time = time.time()
    print("Loading explicitly identified CPU candidate", flush=True)
    if not model_dir.is_absolute():
        model_dir = ROOT / model_dir
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    try:
        candidate = create_pipeline(args, model_dir=model_dir, identity_manifest=manifest_path)
    except BaseException as exc:
        record_failure(result, save, "candidate_load_and_identity", exc)
        raise
    result["model_dir"] = str(candidate.model_dir)
    result["package_identity"] = candidate.package_identity
    result["load_seconds"] = candidate.load_seconds
    save()
    print("Loading qualified CPU reference", flush=True)
    try:
        reference = load_model("cpu")
    except BaseException as exc:
        record_failure(result, save, "reference_load", exc)
        raise

    # Mark this explicitly: the captured output scores must be compared using
    # the native BF16 reference hidden state (the qualified tied head scores
    # BF16 projection results). Recompute each endpoint independently because
    # very short CPU reference inputs can be shape-dependent.
    result["reference_semantics"] = "native_endpoint_hidden_per_matched_prefix"
    save()

    with torch.inference_mode():
        for case in cases:
            ids = case["input_ids"]
            ids_tensor = torch.tensor([ids], dtype=torch.long)
            case_result = {
                "name": case["name"],
                "fixture": case["source"],
                "input_ids": ids,
                "partitions": case["partitions"],
                "matched_prefixes": [],
            }
            result["cases"].append(case_result)
            try:
                request = candidate.request()
            except BaseException as exc:
                record_failure(result, save, f"request_create:{case['name']}", exc)
                raise
            save()
            absolute = 0
            try:
                for part_index, part_length in enumerate(case["partitions"]):
                    start = absolute
                    end = start + part_length
                    if end > len(ids):
                        raise RuntimeError(f"Partition plan exceeds input in {case['name']}.")
                    chunk_hidden: list[np.ndarray] = []

                    def capture(name: str, value: np.ndarray) -> None:
                        if name == "final_hidden":
                            chunk_hidden.append(value.copy())

                    got_scores = candidate.forward(ids[start:end], request, capture)
                    captured_hidden = np.concatenate(chunk_hidden, axis=1)
                    endpoint_hidden = reference.reference_hidden(ids_tensor[:, :end]).detach().cpu()
                    ref_chunk_hidden = endpoint_hidden[:, start:end]
                    hidden = hidden_record(captured_hidden, ref_chunk_hidden, limits)
                    reference_scores = reference.scores(endpoint_hidden[:, -1:])
                    scores = score_record(got_scores, reference_scores, candidate.valid_vocab, limits)
                    record = {
                        "partition_index": part_index,
                        "partition_start": start,
                        "partition_end_exclusive": end,
                        "prefix_length": end,
                        "runtime_internal_chunks": [min(32, part_length - offset) for offset in range(0, part_length, 32)],
                        "hidden": hidden,
                        "scores": scores,
                        "request_length": request.length,
                        "passed": hidden["passed"] and scores["passed"] and request.length == end,
                    }
                    case_result["matched_prefixes"].append(record)
                    absolute = end
                    save()
                    print(case["name"], "prefix", end, "score max_abs", scores["max_abs"], "hidden relRMSE", hidden["relative_rmse"], "pass", record["passed"], flush=True)
            except BaseException as exc:
                case_result["failure"] = {"exception_type": type(exc).__name__, "message": str(exc)}
                record_failure(result, save, f"partition_case:{case['name']}", exc)
                raise
            finally:
                request.close()
                case_result["request_disposed"] = request.closed and not request.states
                save()
            case_result["passed"] = absolute == len(ids) and case_result["request_disposed"] and all(c["passed"] for c in case_result["matched_prefixes"])
            save()

    repeat_case = next(case for case in cases if case["name"] == "single_prefill_33")
    repeat_ids = repeat_case["input_ids"]
    first_request = candidate.request()
    first_hidden: list[np.ndarray] = []
    try:
        first_scores = candidate.forward(repeat_ids, first_request, lambda n, v: first_hidden.append(v.copy()) if n == "final_hidden" else None)
    except BaseException as exc:
        record_failure(result, save, "partition_repeat_first", exc)
        raise
    finally:
        first_request.close()
    second_request = candidate.request()
    second_hidden: list[np.ndarray] = []
    second_scores = None
    second_start = 0
    try:
        for length in [1, 32]:
            second_scores = candidate.forward(repeat_ids[second_start : second_start + length], second_request, lambda n, v: second_hidden.append(v.copy()) if n == "final_hidden" else None)
            second_start += length
    except BaseException as exc:
        record_failure(result, save, "partition_repeat_second", exc)
        raise
    finally:
        second_request.close()
    first_h = np.concatenate(first_hidden, axis=1)
    second_h = np.concatenate(second_hidden, axis=1)
    result["partition_repeat"] = {
        "input_length": len(repeat_ids),
        "first_partitioning": [33],
        "second_partitioning": [1, 32],
        "scores_exact": bool(np.array_equal(first_scores, second_scores)),
        "hidden_exact": bool(np.array_equal(first_h, second_h)),
        "requests_disposed": first_request.closed and not first_request.states and second_request.closed and not second_request.states,
    }
    result["elapsed_seconds"] = time.time() - start_time
    result["status"] = "completed"
    result["numerical_pass"] = all(case["passed"] for case in result["cases"])
    result["partition_repeat_pass"] = result["partition_repeat"]["scores_exact"] and result["partition_repeat"]["hidden_exact"] and result["partition_repeat"]["requests_disposed"]
    result["passed"] = result["numerical_pass"] and result["partition_repeat_pass"]
    save()
    print(json.dumps({"status": result["status"], "passed": result["passed"], "cases": len(result["cases"]), "partition_repeat": result["partition_repeat"], "elapsed_seconds": result["elapsed_seconds"]}, indent=2), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
