"""Inspect pinned inputs without loading weights or executing the model."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sources/runtime/paradigma-inc-limite-violetto-cd68d27"
sys.path.insert(0, str(SOURCE / "src"))

import coremltools as ct
import torch
from transformers import AutoTokenizer
from tokenizers import Tokenizer
from limite_vllm.configuration import LimiteConfig
from limite_vllm.validation import validate


def main() -> None:
    raw = json.loads((ROOT / "sources/model/config.json").read_text())
    config = LimiteConfig.from_dict(raw)
    validate(config)
    assert config.sliding_window == 1025
    assert config.to_dict()["sliding_window"] == 1024
    assert config.lm_head_precision_mode == "oracle_exact"
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "sources/model", local_files_only=True)
    standalone_template = (ROOT / "sources/model/chat_template.jinja").read_text()
    assert tokenizer.chat_template == standalone_template
    raw_tokenizer = Tokenizer.from_file(str(ROOT / "sources/model/tokenizer.json"))
    assert tokenizer.convert_tokens_to_ids("<|endoftext|>") == raw["bos_token_id"]
    assert tokenizer.convert_tokens_to_ids("<|im_end|>") == raw["eos_token_id"]
    cases = []
    for prompt in ("Compute 17 * 23.", "Solve x^2 - 5x + 6 = 0 and give the sum of its roots.", "Find all integer solutions of x^2 = 49."):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True)
        ids = encoded["input_ids"]
        assert tokenizer(text, add_special_tokens=False)["input_ids"] == ids
        assert raw_tokenizer.encode(text, add_special_tokens=False).ids == ids
        assert ids[0] == raw["bos_token_id"]
        assert text.endswith("<|im_start|>assistant\n")
        assert tokenizer.decode(ids, skip_special_tokens=False) == text
        cases.append({"messages": messages, "formatted_prompt": text, "input_ids": ids,
                      "token_count": len(ids), "token_ids_json_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()})
    bf16_eps = torch.finfo(torch.bfloat16).eps
    fp16_eps = torch.finfo(torch.float16).eps
    # Theoretical storage only; no allocation, model load, or memory benchmark.
    kv_bytes = 2 * config.num_hidden_layers * 2048 * config.num_key_value_heads * config.head_dim * 2
    result = {
        "kind": "configuration/tokenizer/package inspection; no neural outputs",
        "python": platform.python_version(), "architecture": platform.machine(),
        "packages": {p: importlib.metadata.version(p) for p in ("torch", "coremltools", "transformers", "numpy", "pytest", "tokenizers", "safetensors")},
        "torch_mps_built": torch.backends.mps.is_built(), "torch_mps_available": torch.backends.mps.is_available(),
        "coremltools_macos15_target_available": hasattr(ct.target, "macOS15"),
        "published_config_validation": "PASS", "serialized_window_distance": 1024,
        "runtime_window_span": config.sliding_window, "head_precision_mode": config.lm_head_precision_mode,
        "bf16_normalization_epsilon": bf16_eps, "fp16_normalization_epsilon": fp16_eps,
        "cache_theoretical_bytes_at_2048_fp16_all_layers": kv_bytes,
        "cache_theoretical_mib_at_2048_fp16_all_layers": kv_bytes / 2**20,
        "cache_note": "K+V only, batch 1, 48 layers, 2 KV heads, 128 head dimension, 2 bytes per scalar; excludes activations, weights, compiler, temporary buffers and runtime copies.",
        "tokenizer_template_matches_standalone": True, "tokenizer_roundtrips": "PASS",
        "raw_tokenizer_agreement_on_cases": "PASS",
        "tokenizer_eos_token_id": tokenizer.eos_token_id,
        "model_generation_eos_token_id": raw["eos_token_id"],
        "stop_token_note": "Use the published model/generation EOS (151645, im_end). Tokenizer metadata EOS is 151643, endoftext, and is different. Do not silently substitute tokenizer.eos_token_id.",
        "cases": cases,
    }
    target = ROOT / "evidence/01-reference/local-input-validation.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2))
    print("Prepared", len(cases), "exact tokenized inputs; no expected neural outputs.")


if __name__ == "__main__":
    main()
