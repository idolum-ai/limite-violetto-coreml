"""Prepare a sampled Core ML arithmetic run with periodic full-prefix audits.

Requires an explicit candidate package root and identity manifest. This is a
bounded release audit; do not run until the short CPU candidate gate passes.
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
from portable.generate import choose_token, tokenizer
from portable.model import load_model
from scripts.probe_full_model import metrics
from scripts.runtime_provenance import runtime_provenance


FROZEN_ACCEPTANCE_SHA256 = "cb4d315266126df4a98194f73f8c10671310717c06ebfd191d3dac2334bf4233"
PERIODIC_AUDIT_INTERVAL = 128
MAX_NEW_TOKENS = 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_output_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("--evidence-dir must be a safe repository-relative path")
    output = (ROOT / path).resolve()
    evidence_root = (ROOT / "evidence/04d-audit").resolve()
    if not output.is_relative_to(evidence_root):
        raise ValueError("--evidence-dir must be under evidence/04d-audit")
    return output


def compare_scores(candidate: np.ndarray, reference: torch.Tensor, valid_vocab: int, tolerance: float) -> dict:
    candidate_valid = candidate.reshape(-1)[:valid_vocab]
    reference_valid = reference.reshape(-1)[:valid_vocab]
    value = metrics(torch.from_numpy(candidate_valid.copy()), reference_valid)
    candidate_top = torch.from_numpy(candidate_valid.copy()).topk(5)
    reference_top = reference_valid.topk(5)
    margin = float(reference_top.values[0] - reference_top.values[1])
    top1_match = int(candidate_top.indices[0]) == int(reference_top.indices[0])
    overlap = len(set(candidate_top.indices.tolist()) & set(reference_top.indices.tolist()))
    ranking_pass = top1_match or margin <= 2 * tolerance
    return {
        **value,
        "top1_candidate": int(candidate_top.indices[0]),
        "top1_reference": int(reference_top.indices[0]),
        "top1_match": top1_match,
        "reference_top1_margin": margin,
        "top5_overlap": overlap,
        "ranking_pass": ranking_pass,
        "passed": value["max_abs"] <= tolerance and ranking_pass,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="Explicit candidate package root.")
    parser.add_argument("--identity-manifest", required=True, help="Identity manifest for that exact candidate root.")
    parser.add_argument("--evidence-dir", required=True, help="New unique directory below evidence/04d-audit/.")
    parser.add_argument("--tiny-prefix-replay", action="store_true", help="Opt into CPU-equivalent cached-prefix replay below 16 tokens.")
    add_loader_arguments(parser)
    args = parser.parse_args()
    output = safe_output_path(args.evidence_dir)
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

    fixture_path = ROOT / "evidence/04-coreml/generation-cpu.json"
    fixture = json.loads(fixture_path.read_text())
    comparison_path = ROOT / "evidence/04-coreml/generation-comparison.json"
    comparison = json.loads(comparison_path.read_text())
    qualified_reference = comparison["reference"]
    if fixture.get("greedy") is not False or fixture.get("seed") != 0:
        raise RuntimeError("Existing arithmetic sample fixture settings changed.")
    if fixture.get("temperature") != 0.6 or fixture.get("top_p") != 0.95:
        raise RuntimeError("Existing arithmetic sample fixture sampler changed.")
    if fixture.get("stop_token_ids") != [151643, 151645]:
        raise RuntimeError("Existing arithmetic sample fixture stop policy changed.")
    prompt_ids = fixture["prompt_ids"]
    settings = {
        "prompt_ids": prompt_ids,
        "greedy": fixture["greedy"],
        "seed": fixture["seed"],
        "temperature": fixture["temperature"],
        "top_p": fixture["top_p"],
        "stop_token_ids": fixture["stop_token_ids"],
    }
    qualified_settings = {
        key: qualified_reference.get(key)
        for key in ["prompt_ids", "greedy", "seed", "temperature", "top_p", "stop_token_ids"]
    }
    if not comparison.get("identical_inputs_and_sampling_settings") or settings != qualified_settings:
        raise RuntimeError("Existing CPU arithmetic fixture does not match the qualified reference inputs and sampling settings.")
    candidate_dir = Path(args.model_dir).expanduser()
    manifest_path = Path(args.identity_manifest).expanduser()
    if not candidate_dir.is_absolute():
        candidate_dir = ROOT / candidate_dir
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path

    hashes = {
        "scripts/audit_coreml_generation.py": sha256(Path(__file__)),
        "coreml_port/runtime.py": sha256(ROOT / "coreml_port/runtime.py"),
        "coreml_port/tiny_prefix_runtime.py": sha256(ROOT / "coreml_port/tiny_prefix_runtime.py"),
        "coreml_port/generate.py": sha256(ROOT / "coreml_port/generate.py"),
        "portable/model.py": sha256(ROOT / "portable/model.py"),
        "portable/generate.py": sha256(ROOT / "portable/generate.py"),
        "evidence/03-portable/ACCEPTANCE.json": acceptance_hash,
        "evidence/03-portable/calibration-cpu.json": sha256(calibration_path),
        "evidence/04-coreml/generation-cpu.json": sha256(fixture_path),
        "evidence/04-coreml/generation-comparison.json": sha256(comparison_path),
        "identity_manifest": sha256(manifest_path),
    }
    result = {
        "status": "starting",
        "compute": "CPU_ONLY",
        "runtime_adapter": runtime_adapter_name(args),
        "loader_provenance": loader_provenance(args),
        "runtime_provenance": runtime_provenance(),
        "scope": "sampled arithmetic generation with periodic full-prefix scores against qualified local CPU reference",
        "thresholds_changed": False,
        "acceptance_sha256": acceptance_hash,
        "limits": limits,
        "source_sha256": hashes,
        "candidate_dir_argument": str(candidate_dir),
        "identity_manifest_argument": str(manifest_path),
        "qualified_reference_settings": settings,
        "qualified_reference_generated_ids_sha256": hashlib.sha256(
            json.dumps(qualified_reference["generated_ids"], separators=(",", ":")).encode()
        ).hexdigest(),
        "fixture": {
            "path": str(fixture_path.relative_to(ROOT)),
            "prompt_ids": prompt_ids,
            "sampling": {"greedy": False, "seed": 0, "temperature": 0.6, "top_p": 0.95},
            "published_eos": 151645,
            "explicit_tokenizer_eos": 151643,
            "max_new_tokens": MAX_NEW_TOKENS,
            "periodic_audit_interval": PERIODIC_AUDIT_INTERVAL,
        },
        "audits": [],
        "generated_ids": [],
        "steps": [],
    }
    result_path = output / "results.json"

    def save() -> None:
        temporary = output / "results.json.tmp"
        temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        temporary.replace(result_path)

    save()
    torch.set_num_threads(4)
    started = time.time()
    print("Loading explicitly identified CPU candidate", flush=True)
    try:
        candidate = create_pipeline(args, model_dir=candidate_dir, identity_manifest=manifest_path)
    except BaseException as exc:
        result.update(status="failed", passed=False, failure={"stage": "candidate_load_and_identity", "exception_type": type(exc).__name__, "message": str(exc)})
        save()
        raise
    try:
        if candidate.eos != 151645 or candidate.valid_vocab != 151667:
            raise RuntimeError("Candidate EOS or valid vocabulary differs from the frozen generation fixture.")
        if len(prompt_ids) + MAX_NEW_TOKENS > candidate.capacity:
            raise RuntimeError("Arithmetic fixture exceeds the configured request context.")
        if any(type(token) is not int or token < 0 or token >= candidate.valid_vocab for token in prompt_ids):
            raise RuntimeError("Fixture contains an invalid token ID.")
    except BaseException as exc:
        result.update(status="failed", passed=False, failure={"stage": "candidate_contract", "exception_type": type(exc).__name__, "message": str(exc)})
        save()
        raise
    result["candidate_dir"] = str(candidate.model_dir)
    result["package_identity"] = candidate.package_identity
    result["load_seconds"] = candidate.load_seconds
    save()
    print("Loading qualified CPU reference", flush=True)
    try:
        reference = load_model("cpu")
    except BaseException as exc:
        result.update(status="failed", passed=False, failure={"stage": "reference_load", "exception_type": type(exc).__name__, "message": str(exc)})
        save()
        raise
    try:
        tok = tokenizer()
        if tok.eos_token_id != 151643:
            raise RuntimeError("Tokenizer EOS differs from the explicit fixture stop ID.")
    except BaseException as exc:
        result.update(status="failed", passed=False, failure={"stage": "tokenizer_contract", "exception_type": type(exc).__name__, "message": str(exc)})
        save()
        raise

    generator = torch.Generator(device="cpu").manual_seed(0)
    try:
        request = candidate.request()
    except BaseException as exc:
        result.update(status="failed", passed=False, failure={"stage": "request_create", "exception_type": type(exc).__name__, "message": str(exc), "request_disposed": None})
        save()
        raise
    generated: list[int] = []
    scores_checked = None
    scored_prefix: list[int] | None = None
    finish_reason = "length"
    audit_errors: list[float] = []

    def audit(label: str, prefix: list[int], candidate_scores: np.ndarray) -> None:
        ref_ids = torch.tensor([prefix], dtype=torch.long, device=reference.device)
        ref_hidden = reference.reference_hidden(ref_ids)
        ref_scores = reference.scores(ref_hidden[:, -1:])
        record = {
            "label": label,
            "generated_tokens_before_next_token": len(generated) if label != "terminal" else max(0, len(generated) - 1),
            "prefix_length": len(prefix),
            "scores": compare_scores(candidate_scores, ref_scores, candidate.valid_vocab, limits["score_max_abs"]),
        }
        result["audits"].append(record)
        audit_errors.append(record["scores"]["max_abs"])
        save()
        print(label, "prefix", len(prefix), "score max_abs", record["scores"]["max_abs"], "pass", record["scores"]["passed"], flush=True)

    stage = "generation"
    try:
        for step in range(MAX_NEW_TOKENS):
            prefix = prompt_ids + generated
            start = time.perf_counter()
            scores_checked = candidate.forward(prefix if step == 0 else [generated[-1]], request)
            scored_prefix = list(prefix)
            seconds = time.perf_counter() - start
            if step % PERIODIC_AUDIT_INTERVAL == 0:
                audit(f"periodic_{step}", prefix, scores_checked)

            token = choose_token(torch.from_numpy(scores_checked), candidate.valid_vocab, False, generator, temperature=0.6, top_p=0.95)
            generated.append(token)
            result["generated_ids"] = generated
            result["steps"].append({"index": step, "token": token, "seconds": seconds})
            if token == candidate.eos:
                finish_reason = "eos"
            elif token == 151643:
                finish_reason = "stop_token"
            save()
            if finish_reason != "length":
                break

        if scores_checked is None or scored_prefix is None:
            raise RuntimeError("Generation produced no score step.")
        audit("terminal", scored_prefix, scores_checked)
    except BaseException as exc:
        result.update(status="failed", passed=False, failure={"stage": stage, "exception_type": type(exc).__name__, "message": str(exc)})
        save()
        raise
    finally:
        request.close()
        result["request_disposed_after_run"] = request.closed and not request.states
        save()

    text = tok.decode(generated, skip_special_tokens=True)
    think_marker_present = "</think>" in text
    boxed_answer = think_marker_present and "\\boxed{391}" in text.split("</think>", 1)[1]
    common = 0
    frozen_tokens = fixture["generated_ids"]
    for actual, prior in zip(generated, frozen_tokens):
        if actual != prior:
            break
        common += 1
    qualified_common = 0
    for actual, prior in zip(generated, qualified_reference["generated_ids"]):
        if actual != prior:
            break
        qualified_common += 1
    result["generation"] = {
        "generated_tokens": len(generated),
        "finish_reason": finish_reason,
        "complete": finish_reason in {"eos", "stop_token"},
        "stop_token_ids": [151643, candidate.eos],
        "completion_policy": "published_eos_plus_explicit_tokenizer_eos",
        "greedy": False,
        "seed": 0,
        "temperature": 0.6,
        "top_p": 0.95,
        "text": text,
        "boxed_391_after_think": boxed_answer,
        "think_marker_present": think_marker_present,
        "qualified_reference_settings_match": settings == qualified_settings,
        "common_prefix_with_qualified_cpu_tokens": qualified_common,
        "qualified_cpu_token_count": len(qualified_reference["generated_ids"]),
        "common_prefix_with_existing_candidate_tokens": common,
        "existing_candidate_token_count": len(frozen_tokens),
        "request_disposed": request.closed and not request.states,
    }
    result["elapsed_seconds"] = time.time() - started
    result["max_audited_score_error"] = max(audit_errors) if audit_errors else None
    result["audits_pass"] = bool(result["audits"]) and all(audit["scores"]["passed"] for audit in result["audits"])
    result["generation_pass"] = result["generation"]["complete"] and boxed_answer and result["generation"]["request_disposed"] and result["generation"]["qualified_reference_settings_match"]
    result["passed"] = result["audits_pass"] and result["generation_pass"]
    result["status"] = "completed"
    save()
    print(json.dumps({"status": result["status"], "passed": result["passed"], "audits_pass": result["audits_pass"], "generation": {k: result["generation"][k] for k in ["generated_tokens", "finish_reason", "complete", "boxed_391_after_think", "common_prefix_with_existing_candidate_tokens"]}, "elapsed_seconds": result["elapsed_seconds"]}, indent=2), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
