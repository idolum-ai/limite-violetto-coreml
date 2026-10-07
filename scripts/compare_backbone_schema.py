"""Compare published architectures on meta tensors; no weights or inference."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "sources/runtime/paradigma-inc-limite-violetto-cd68d27"
sys.path[:0] = [str(ROOT / "sources/value-model"), str(RUNTIME / "src")]
import pytest
import torch
from configuration_gmpt import GMPTConfig
from modeling_backbone import GMPTModel
from limite_vllm.configuration import LimiteConfig


def main() -> None:
    actor = json.loads((ROOT / "sources/model/config.json").read_text())
    critic = json.loads((ROOT / "sources/value-model/backbone_config.json").read_text())
    same = sorted(k for k in actor.keys() & critic.keys() if actor[k] == critic[k])
    differences = {k: {"violetto": actor.get(k), "value_model": critic.get(k)}
                   for k in sorted(actor.keys() | critic.keys()) if actor.get(k) != critic.get(k)}
    spec = importlib.util.spec_from_file_location("upstream_contract", RUNTIME / "tests/test_package_contract.py")
    test_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_module)
    with pytest.MonkeyPatch.context() as patch:
        test_module._install_vllm_import_stubs(patch)
        from limite_vllm.modeling import LimiteForCausalLM
        with torch.device("meta"):
            backbone = GMPTModel(GMPTConfig(**critic))
            cfg = LimiteConfig.from_dict(actor)
            runtime = LimiteForCausalLM(vllm_config=SimpleNamespace(
                model_config=SimpleNamespace(hf_config=cfg),
                cache_config=SimpleNamespace(sliding_window=None)))
    expected = {}
    for name, param in runtime.named_parameters():
        if name.endswith(".self_attn.qkv_proj.weight"):
            stem = name.removesuffix("qkv_proj.weight")
            for projection, rows in [("q", cfg.hidden_size), ("k", cfg.num_key_value_heads * cfg.head_dim), ("v", cfg.num_key_value_heads * cfg.head_dim)]:
                expected[stem + projection + "_proj.weight"] = [rows, cfg.hidden_size]
        elif name.endswith(".mlp.gate_up_proj.weight"):
            stem = name.removesuffix("gate_up_proj.weight")
            for projection in ("gate", "up"):
                expected[stem + projection + "_proj.weight"] = [cfg.intermediate_size, cfg.hidden_size]
        else:
            expected[name] = list(param.shape)
    scales = {"qkv": "self_attn.qkv_scale", "o": "self_attn.o_scale",
              "resid_attn": "resid_lambda_attn", "post_attn": "post_lambda_attn",
              "resid_mlp": "resid_lambda_mlp", "post_mlp": "post_lambda_mlp"}
    mapping = []
    for name, param in backbone.named_parameters():
        target = "model." + name
        match = re.fullmatch(r"layers\.(\d+)\.scales\.(.+)", name)
        scalar = match is not None
        if scalar:
            target = f"model.layers.{match[1]}.{scales[match[2]]}"
        adapted_shape = [] if scalar else list(param.shape)
        mapping.append({"backbone_name": name, "runtime_checkpoint_name": target,
                        "backbone_shape": list(param.shape), "adapted_shape": adapted_shape,
                        "runtime_expected_shape": expected.get(target),
                        "matches": adapted_shape == expected.get(target)})
    mapped = {item["runtime_checkpoint_name"] for item in mapping}
    result = {"kind": "meta-device architecture/parameter-schema comparison; no checkpoint tensors or inference",
              "shared_equal_config_fields": same, "shared_equal_config_count": len(same),
              "config_differences": differences, "mapping": mapping,
              "backbone_parameter_tensors": len(mapping), "runtime_source_parameter_tensors": len(expected),
              "unmapped_runtime_parameters": sorted(set(expected) - mapped),
              "unexpected_backbone_parameters": sorted(mapped - set(expected)),
              "scalar_reshapes": sum(1 for item in mapping if item["backbone_shape"] != item["adapted_shape"]),
              "schema_matches": all(item["matches"] for item in mapping) and mapped == set(expected),
              "weight_contents_validated": False, "numerical_fidelity_validated": False,
              "limitations": ["Shape/name agreement is inferred from constructors and loader source, not the downloaded Violetto checkpoint.",
                              "Value model weights and scalar head are different; do not reuse them as Violetto outputs.",
                              "Value loader casts every backbone parameter to BF16; Violetto keeps selected parameters in FP32, including XSA alpha. Preserve original checkpoint dtypes when adapting.",
                              "Backbone has no causal LM head, cache, or generation loop. Violetto head, score transform, prompt, and EOS must be supplied explicitly."]}
    path = ROOT / "evidence/01-reference/backbone-schema-comparison.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"mapping", "shared_equal_config_fields", "config_differences"}}, indent=2))
    if not result["schema_matches"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
