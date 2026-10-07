"""Run three fixed, objectively scored practical tasks on the portable CPU model.

Use --self-check for parser checks without loading model weights or running
inference. A normal run reuses one verified model and writes atomic per-token
progress to a fresh evidence/05-practical/<attempt>/results.json.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
MAX_NEW_TOKENS = 1024
CAPACITY = 2048
SEED = 0
TEMPERATURE = 0.6
TOP_P = 0.95
STOP_IDS = (151643, 151645)
TASKS = (
    {"id": "arithmetic", "prompt": "Compute 28 * 17.", "expected": "476"},
    {"id": "simple_algebra", "prompt": "Solve 4x + 3 = 13. What is x?", "expected": "5/2"},
    {"id": "short_word_problem", "prompt":
     "Maya has 18 marbles. She gives away one third of them, then buys 5 more. How many marbles does she have now?",
     "expected": "17"},
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _boxed_bodies(final: str) -> tuple[list[str], str | None]:
    marker = r"\boxed{"
    bodies = []
    start = 0
    while (at := final.find(marker, start)) != -1:
        depth = 1
        i = at + len(marker)
        body_start = i
        while i < len(final) and depth:
            if final[i] == "{":
                depth += 1
            elif final[i] == "}":
                depth -= 1
            i += 1
        if depth:
            return [], "malformed_box"
        bodies.append(final[body_start:i - 1])
        start = i
    return bodies, None


def normalize_numeric(body: str) -> Fraction | None:
    """Accept only a simple signed integer, decimal, or fraction."""
    value = re.sub(r"\s+", "", body)
    value = value.replace(r"\,", "").replace(r"\!", "")
    match = re.fullmatch(r"([+-]?)\\(?:dfrac|frac)\{([+-]?\d+)\}\{([+-]?\d+)\}", value)
    if match:
        sign, numerator, denominator = match.groups()
        if int(denominator) == 0:
            return None
        return Fraction((-1 if sign == "-" else 1) * int(numerator), int(denominator))
    match = re.fullmatch(r"([+-]?\d+)/([+-]?\d+)", value)
    if match:
        numerator, denominator = map(int, match.groups())
        return None if denominator == 0 else Fraction(numerator, denominator)
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", value):
        try:
            return Fraction(Decimal(value))
        except (InvalidOperation, ValueError):
            return None
    return None


def parse_final_box(text: str) -> dict:
    if "</think>" not in text:
        return {"status": "missing_think_close"}
    final = text.rsplit("</think>", 1)[1]
    bodies, error = _boxed_bodies(final)
    if error:
        return {"status": error}
    if not bodies:
        return {"status": "missing_final_box"}
    if len(bodies) != 1:
        return {"status": "multiple_final_boxes", "count": len(bodies)}
    normalized = normalize_numeric(bodies[0])
    if normalized is None:
        return {"status": "nonnumeric_final_box", "body": bodies[0]}
    return {"status": "parsed", "body": bodies[0], "normalized": str(normalized)}


def self_check() -> None:
    checks = (
        (r"reason \boxed{999}</think> Final: \boxed{476}.", "parsed", "476"),
        (r"</think>\boxed{\frac{10}{4}}", "parsed", "5/2"),
        (r"</think>\boxed{-\dfrac{3}{2}}", "parsed", "-3/2"),
        (r"</think>\boxed{2.50}", "parsed", "5/2"),
        (r"\boxed{476}", "missing_think_close", None),
        (r"</think>The answer is 476.", "missing_final_box", None),
        (r"</think>\boxed{5} and \boxed{6}", "multiple_final_boxes", None),
        (r"</think>\boxed{five}", "nonnumeric_final_box", None),
        (r"</think>\boxed{1/0}", "nonnumeric_final_box", None),
        (r"</think>\boxed{\frac{5}{2}", "malformed_box", None),
    )
    for text, status, answer in checks:
        got = parse_final_box(text)
        assert got["status"] == status, (text, got)
        if answer is not None:
            assert got["normalized"] == answer, (text, got)
    assert tuple(Fraction(task["expected"]) for task in TASKS) == (
        Fraction(476), Fraction(5, 2), Fraction(17))
    print(json.dumps({"self_check": "pass", "parser_cases": len(checks),
                      "task_count": len(TASKS)}))


def prepare(tokenizer) -> list[dict]:
    template = (ROOT / "sources/model/chat_template.jinja").read_text()
    if tokenizer.chat_template != template:
        raise ValueError("Tokenizer template differs from pinned chat_template.jinja")
    if tokenizer.eos_token_id != 151643:
        raise ValueError("Unexpected tokenizer EOS")
    prepared = []
    for task in TASKS:
        messages = [{"role": "user", "content": task["prompt"]}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer.apply_chat_template(messages, tokenize=True,
                                                add_generation_prompt=True, return_dict=True)
        ids = encoded["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        if tokenizer.decode(ids, skip_special_tokens=False) != formatted:
            raise ValueError(f"Template/ID roundtrip failed: {task['id']}")
        if not (0 < len(ids) and len(ids) + MAX_NEW_TOKENS <= CAPACITY):
            raise ValueError(f"Context budget exceeded: {task['id']}")
        prepared.append({"id": task["id"], "messages": messages, "prompt": task["prompt"],
                         "formatted_prompt": formatted, "prompt_ids": ids,
                         "prompt_tokens": len(ids), "expected": task["expected"],
                         "expected_normalized": str(Fraction(task["expected"])),
                         "status": "pending"})
    return prepared


def run(attempt: str) -> int:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", attempt):
        raise ValueError("Attempt must be a lowercase path-safe slug")
    out = ROOT / "evidence/05-practical" / attempt
    out.mkdir(parents=True, exist_ok=False)
    target = out / "results.json"
    sys.path.insert(0, str(ROOT))
    import torch
    from portable.generate import generate, tokenizer as load_tokenizer
    from portable.model import load_model

    torch.set_num_threads(4)
    result = {"status": "preparing", "attempt": attempt, "device": "cpu",
              "threads": torch.get_num_threads(),
              "settings": {"max_new_tokens": MAX_NEW_TOKENS, "seed": SEED,
                           "greedy": False, "temperature": TEMPERATURE, "top_p": TOP_P,
                           "stop_token_ids": list(STOP_IDS), "capacity": CAPACITY,
                           "completion_required_for_pass": True,
                           "prompt_template": "original pinned chat_template.jinja"},
              "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in (
                  Path(__file__), ROOT / "portable/model.py", ROOT / "portable/generate.py",
                  ROOT / "sources/model/config.json", ROOT / "sources/model/tokenizer.json",
                  ROOT / "sources/model/tokenizer_config.json",
                  ROOT / "sources/model/chat_template.jinja",
                  ROOT / "sources/model/generation_config.json", ROOT / "manifest.json",
                  ROOT / "evidence/05-practical/task-design.md")},
              "tasks": []}
    atomic_json(target, result)
    try:
        tokenizer = load_tokenizer()
        result["tasks"] = prepare(tokenizer)
        result["status"] = "loading_model"
        atomic_json(target, result)
        model = load_model("cpu")
        if model.capacity != CAPACITY or model.config.eos_token_id != 151645:
            raise ValueError("Unexpected model capacity or published EOS")
    except BaseException as error:
        result["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "error"
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        atomic_json(target, result)
        raise
    result["model_identity"] = {"checkpoint_sha256": model.load_report["checkpoint_sha256"],
                                "hash_verified_this_load": model.load_report["hash_verified_this_load"],
                                "parameter_count": model.load_report["parameter_count"],
                                "attention_min_query_rows": model.attention_min_query_rows}
    result["status"] = "running"
    atomic_json(target, result)
    for task in result["tasks"]:
        task["status"] = "generating"
        task["generated_ids"] = []
        task["started_monotonic_seconds"] = time.perf_counter()
        atomic_json(target, result)

        def on_token(token: int) -> None:
            task["generated_ids"].append(token)
            task["generated_tokens"] = len(task["generated_ids"])
            task["progress_elapsed_seconds"] = time.perf_counter() - task["started_monotonic_seconds"]
            atomic_json(target, result)

        try:
            generation = generate(model, task["prompt_ids"], MAX_NEW_TOKENS,
                                  greedy=False, seed=SEED, on_token=on_token,
                                  additional_stop_ids=(151643,))
            if generation["generated_ids"] != task["generated_ids"]:
                raise RuntimeError("Token callback diverged from generation result")
            text = tokenizer.decode(generation["generated_ids"], skip_special_tokens=True)
            parsed = parse_final_box(text)
            task["generation"] = generation
            task["text"] = text
            task["parsed_final_answer"] = parsed
            task["passed"] = bool(generation["complete"] and
                                  parsed.get("status") == "parsed" and
                                  parsed["normalized"] == task["expected_normalized"])
            task["status"] = "complete"
            task["wall_seconds"] = time.perf_counter() - task["started_monotonic_seconds"]
            atomic_json(target, result)
            print(json.dumps({"task": task["id"], "complete": generation["complete"],
                              "generated_tokens": generation["generated_tokens"],
                              "parsed": parsed, "passed": task["passed"]}), flush=True)
        except BaseException as error:
            # portable.generate.generate resets its cache in a finally block,
            # including when this callback or the generation loop raises.
            task["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "error"
            task["error"] = {"type": type(error).__name__, "message": str(error)}
            task["text"] = tokenizer.decode(task["generated_ids"], skip_special_tokens=True)
            task["wall_seconds"] = time.perf_counter() - task["started_monotonic_seconds"]
            result["status"] = task["status"]
            atomic_json(target, result)
            raise
    result["status"] = "complete"
    result["all_passed"] = all(task["passed"] for task in result["tasks"])
    result["suite_note"] = "Three descriptive local CPU-reference tasks; not a general quality certification. Per-token fsync progress affects wall time."
    atomic_json(target, result)
    print(json.dumps({"status": result["status"], "all_passed": result["all_passed"],
                      "tasks": len(result["tasks"]), "results": str(target)}), flush=True)
    return 0 if result["all_passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", help="Fresh evidence/05-practical/<attempt> directory")
    parser.add_argument("--self-check", action="store_true", help="Parser checks only; no model load or inference")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return 0
    if not args.attempt:
        parser.error("--attempt is required for a generation run")
    return run(args.attempt)


if __name__ == "__main__":
    raise SystemExit(main())
