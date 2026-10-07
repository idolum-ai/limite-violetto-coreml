"""Compare the composed ordered Torch graph with the qualified CPU reference.

This is a numerical audit, not a Core ML conversion or an original-runtime
fidelity claim. Each cache/softmax option names an explicit ablation.
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

import torch

import coreml_port.graphs as graphs
from coreml_port import ordered_attention, ordered_linear
from portable.model import load_model
from scripts.probe_full_model import metrics


GROUP_END_LAYERS = tuple(range(3, 44, 4))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_hashes() -> dict[str, str]:
    names = [
        "portable/model.py",
        "portable/generate.py",
        "coreml_port/graphs.py",
        "coreml_port/ordered_linear.py",
        "coreml_port/ordered_attention.py",
        "coreml_port/source_exp.py",
        "coreml_port/ordered_norm.py",
        "coreml_port/ordered_xsa.py",
        "coreml_port/bf16_gelu.py",
        "scripts/audit_torch_candidate.py",
        "scripts/probe_full_model.py",
        "evidence/03-portable/ACCEPTANCE.json",
        "evidence/03-portable/calibration-cpu.json",
        "evidence/01-reference/local-input-validation.json",
    ]
    return {name: sha256(ROOT / name) for name in names}


def compare(a: torch.Tensor, b: torch.Tensor, kind: str) -> dict:
    value = metrics(a.detach().float().cpu(), b.detach().float().cpu())
    value["kind"] = kind
    return value


def reference_prefix(model, ids: list[int]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    captures: dict[str, torch.Tensor] = {}
    handles = []
    for layer_index, layer in enumerate(model.backbone.layers):
        if layer_index in GROUP_END_LAYERS:
            def keep(module, args, output, index=layer_index):
                value = output[0] if isinstance(output, tuple) else output
                captures[f"layer_{index}"] = value.detach().float().cpu().clone()
            handles.append(layer.register_forward_hook(keep))
    try:
        tensor = torch.tensor([ids], dtype=torch.long, device=model.device)
        hidden = model.reference_hidden(tensor)
        captures["final_hidden"] = hidden.detach().float().cpu().clone()
        return hidden, captures
    finally:
        for handle in handles:
            handle.remove()


def make_candidate(model, cache_dtype: str, crop_cache: bool, softmax: str, norm_mode: str, xsa_mode: str, gelu_mode: str, exponential_mode: str):
    graphs.linear = ordered_linear.linear
    if softmax == "ordered":
        graphs.softmax = ordered_attention.softmax
    else:
        graphs.softmax = lambda x: x.softmax(-1)
    ordered_attention.exponential = torch.exp
    if exponential_mode == "source":
        from coreml_port.source_exp import exp

        ordered_attention.exponential = exp
    if norm_mode == "ordered":
        from coreml_port.ordered_norm import raw_norm

        def ordered_graph_norm(value):
            divisor = torch.tensor([float(value.shape[-1])], dtype=value.dtype, device=value.device)
            return graphs.Q(raw_norm(value, divisor))

        graphs.norm = ordered_graph_norm
    if xsa_mode == "ordered":
        from coreml_port.ordered_xsa import ordered_xsa_dot, ordered_xsa_normalize

        graphs.xsa_normalize = ordered_xsa_normalize
        graphs.xsa_dot = ordered_xsa_dot
    if gelu_mode == "lookup":
        from coreml_port.bf16_gelu import BF16Gelu

        graphs.gelu_factory = BF16Gelu
    embed = graphs.Embedding(model).eval()
    groups = [graphs.Group(model, start).eval() for start in range(0, 48, 4)]
    head = graphs.Head(model).eval()
    if cache_dtype == "fp32":
        for group in groups:
            for name, value in list(group.named_buffers()):
                if name.startswith(("k_cache_", "v_cache_")):
                    setattr(group, name, torch.zeros_like(value, dtype=torch.float32))
    if crop_cache:
        original = graphs.Layer.finish

        def finish_cropped(self, x, residual, q, keys, values, v, positions, slots):
            end = int(positions[-1].item()) + 1
            return original(self, x, residual, q, keys[:, :end], values[:, :end], v, positions, slots[:end])

        graphs.Layer.finish = finish_cropped
    return embed, groups, head


def candidate_forward(embed, groups, head, ids: list[int], start: int, cache_dtype: str):
    ids_tensor = torch.tensor([ids], dtype=torch.long)
    positions = torch.arange(start, start + len(ids), dtype=torch.int32)
    x, ve = embed(ids_tensor)
    tap0 = x
    tap12 = x
    tap23 = x
    boundaries = {}
    for group_index, group in enumerate(groups):
        x, tap23 = group(x, ve, positions, tap0, tap12, tap23)
        boundaries[f"layer_{group_index * 4 + 3}"] = x.detach().float().cpu().clone()
        if group_index == 2:
            tap12 = x
    scores = head(x[:, -1:])
    return x, scores, boundaries


def score_metrics(candidate: torch.Tensor, reference: torch.Tensor, valid_vocab: int) -> dict:
    a = candidate.reshape(-1)[:valid_vocab].float().cpu()
    b = reference.reshape(-1)[:valid_vocab].float().cpu()
    value = compare(a, b, "scores")
    value["top1_match"] = int(a.argmax()) == int(b.argmax())
    top = b.topk(5)
    value["reference_top1_margin"] = float(top.values[0] - top.values[1])
    value["top5_overlap"] = len(set(a.topk(5).indices.tolist()) & set(top.indices.tolist()))
    value["passed"] = value["max_abs"] <= 0.04 and (value["top1_match"] or value["reference_top1_margin"] <= 0.08)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True, help="Unique output subdirectory under evidence/04d-audit/torch-candidate.")
    parser.add_argument("--cache-dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--crop-cache", action="store_true", help="Expose only keys through the current absolute position.")
    parser.add_argument("--softmax", choices=("ordered", "native"), default="ordered")
    parser.add_argument("--exponential", choices=("native", "source"), default="native")
    parser.add_argument("--norm", choices=("native", "ordered"), default="native")
    parser.add_argument("--xsa", choices=("native", "ordered"), default="native")
    parser.add_argument("--gelu", choices=("native", "lookup"), default="native")
    parser.add_argument("--first-case-only", action="store_true")
    args = parser.parse_args()
    if not args.attempt or Path(args.attempt).name != args.attempt:
        parser.error("--attempt must be one path component")

    torch.set_num_threads(4)
    output = ROOT / "evidence/04d-audit/torch-candidate" / args.attempt
    output.mkdir(parents=True, exist_ok=False)
    calibration = json.loads((ROOT / "evidence/03-portable/calibration-cpu.json").read_text())
    limits = calibration["frozen_limits"]
    if not limits.get("calibration_valid"):
        raise RuntimeError("Frozen CPU reference calibration is invalid; refusing candidate comparison.")

    result = {
        "status": "running",
        "scope": "pure Torch ordered graph versus qualified local CPU reference",
        "device": "cpu",
        "cache_dtype": args.cache_dtype,
        "cache_cropped_to_current_position": args.crop_cache,
        "softmax": args.softmax,
        "exponential": args.exponential,
        "norm": args.norm,
        "xsa": args.xsa,
        "gelu": args.gelu,
        "chunk_size": 32,
        "thresholds_changed": False,
        "acceptance_sha256": sha256(ROOT / "evidence/03-portable/ACCEPTANCE.json"),
        "limits": limits,
        "source_sha256": file_hashes(),
        "cases": [],
    }
    path = output / "results.json"

    def save():
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    save()
    start_time = time.time()
    print("Loading qualified CPU reference", flush=True)
    reference = load_model("cpu")
    cases = json.loads((ROOT / "evidence/01-reference/local-input-validation.json").read_text())["cases"]
    selected = cases[:1] if args.first_case_only else cases
    candidate = make_candidate(reference, args.cache_dtype, args.crop_cache, args.softmax, args.norm, args.xsa, args.gelu, args.exponential)
    embed, groups, head = candidate

    with torch.inference_mode():
        for index, case in enumerate(selected):
            for group in groups:
                for name, value in group.named_buffers():
                    if name.startswith(("k_cache_", "v_cache_")):
                        value.zero_()
            prompt = list(case["input_ids"])
            ref_hidden, ref_layers = reference_prefix(reference, prompt)
            case_result = {"index": index, "prompt_length": len(prompt), "prompt_ids": prompt, "prefill_chunks": [], "decode": []}
            position = 0
            for chunk_start in range(0, len(prompt), 32):
                chunk = prompt[chunk_start : chunk_start + 32]
                cand_hidden, cand_scores, boundaries = candidate_forward(embed, groups, head, chunk, position, args.cache_dtype)
                chunk_end = chunk_start + len(chunk)
                chunk_metrics = {
                    "start": chunk_start,
                    "end": chunk_end,
                    "final_hidden": compare(cand_hidden, ref_hidden[:, chunk_start:chunk_end], "hidden"),
                    "group_boundaries": {},
                }
                chunk_metrics["final_hidden"]["passed"] = chunk_metrics["final_hidden"]["relative_rmse"] <= limits["hidden_relative_rmse"] and chunk_metrics["final_hidden"]["cosine"] >= limits["hidden_min_cosine"]
                for name, values in boundaries.items():
                    ref_values = ref_layers.get(name)
                    if ref_values is not None:
                        metric = compare(values, ref_values[:, chunk_start:chunk_end], "group_hidden")
                        metric["passed"] = metric["relative_rmse"] <= limits["hidden_relative_rmse"] and metric["cosine"] >= limits["hidden_min_cosine"]
                        chunk_metrics["group_boundaries"][name] = metric
                case_result["prefill_chunks"].append(chunk_metrics)
                position = chunk_end
                save()

            ref_scores = reference.scores(ref_hidden[:, -1:])
            candidate_scores = cand_scores
            prompt_score = score_metrics(candidate_scores, ref_scores, reference.config.tokenizer_vocab_size)
            case_result["prompt_scores"] = {"prefix_length": len(prompt), **prompt_score}
            case_result["prefix_scores_passed"] = prompt_score["passed"]
            save()

            teacher_tokens = []
            for step in range(4):
                token = int(ref_scores.reshape(-1)[: reference.config.tokenizer_vocab_size].argmax())
                teacher_tokens.append(token)
                full_prefix = prompt + teacher_tokens
                ref_hidden, ref_layers = reference_prefix(reference, full_prefix)
                chunk_hidden, candidate_scores, boundaries = candidate_forward(embed, groups, head, [token], position, args.cache_dtype)
                ref_scores = reference.scores(ref_hidden[:, -1:])
                score_result = score_metrics(candidate_scores, ref_scores, reference.config.tokenizer_vocab_size)
                hidden_result = compare(chunk_hidden, ref_hidden[:, -1:], "hidden")
                hidden_result["passed"] = hidden_result["relative_rmse"] <= limits["hidden_relative_rmse"] and hidden_result["cosine"] >= limits["hidden_min_cosine"]
                case_result["decode"].append({"step": step, "input_token": token, "prefix_length": len(full_prefix), "scores": score_result, "final_hidden": hidden_result})
                position += 1
                save()
            case_result["teacher_forced_tokens"] = teacher_tokens
            case_result["passed"] = (
                case_result["prefix_scores_passed"]
                and all(chunk["final_hidden"]["passed"] and all(m["passed"] for m in chunk["group_boundaries"].values()) for chunk in case_result["prefill_chunks"])
                and all(step["scores"]["passed"] and step["final_hidden"]["passed"] for step in case_result["decode"])
            )
            result["cases"].append(case_result)
            save()
            print(json.dumps({"case": index, "prompt_length": len(prompt), "pass": case_result["passed"], "prompt_score_max_abs": prompt_score["max_abs"], "decode_score_max_abs": [x["scores"]["max_abs"] for x in case_result["decode"]]}, indent=2), flush=True)
            if args.first_case_only:
                break

    result["elapsed_seconds"] = time.time() - start_time
    result["passed"] = len(result["cases"]) == len(selected) and all(case["passed"] for case in result["cases"])
    result["status"] = "complete"
    save()
    print(json.dumps({"status": result["status"], "passed": result["passed"], "attempt": args.attempt, "cases": len(result["cases"]), "elapsed_seconds": result["elapsed_seconds"]}, indent=2), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
