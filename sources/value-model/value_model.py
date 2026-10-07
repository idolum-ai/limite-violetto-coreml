"""Load and score the released value model.

The inference API supports one unpadded sequence at a time and returns
pre-action values for prompt and response token IDs.
"""
from pathlib import Path
import json
from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file

from configuration_gmpt import GMPTConfig
from modeling_backbone import GMPTModel, configure_attention


class ValueModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = GMPTModel(config)
        self.value_head = nn.Linear(config.hidden_size, 1, dtype=torch.float32)

    @property
    def device(self):
        return self.value_head.weight.device

    @torch.no_grad()
    def forward(self, input_ids):
        """Return raw post-token values [1, L] for a single unpadded input."""
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("Expected one unpadded sequence, shape [1, L].")
        if input_ids.dtype != torch.long:
            raise ValueError("input_ids must be torch.int64.")
        n = input_ids.shape[1]
        if not 0 < n <= self.config.max_position_embeddings:
            raise ValueError("Input length is empty or exceeds the context; no automatic truncation.")
        if self.config._attn_implementation == "eager" and n > 4096:
            raise ValueError("The eager reference backend is limited to 4096 tokens; use FlashAttention for longer inputs.")
        ids = input_ids.to(self.device)
        positions = torch.arange(n, device=self.device, dtype=torch.long)[None, :]
        with torch.autocast(device_type=self.device.type, enabled=False):
            h = self.backbone(input_ids=ids, position_ids=positions).last_hidden_state
            values = F.linear(h.float(), self.value_head.weight, self.value_head.bias).squeeze(-1)
        return values

    @torch.no_grad()
    def score_tokens(self, prompt_ids, response_ids):
        """Score original token IDs; continuation values precede each action."""
        for name, tokens in (("prompt_ids", prompt_ids), ("response_ids", response_ids)):
            if not isinstance(tokens, (list, tuple)) or not tokens:
                raise ValueError(f"{name} must be a nonempty list or tuple of integer token IDs.")
            if any(isinstance(t, bool) or not isinstance(t, Integral) for t in tokens):
                raise ValueError(f"{name} must contain integer token IDs; no implicit conversion.")
        ids = torch.tensor([list(prompt_ids) + list(response_ids)], dtype=torch.long, device=self.device)
        post = self(ids)
        p, length = len(prompt_ids), len(response_ids)
        pre = post[0, p - 1:p + length - 1]
        if not torch.isfinite(post).all():
            raise RuntimeError("Non-finite value output.")
        return {
            "prompt_tokens": p,
            "response_tokens": length,
            "pre_action_values": pre.cpu().tolist(),
            "prompt_value": pre[0].item(),
            "mean_response_pre_action_value": pre.mean().item(),
            "last_pre_action_value": pre[-1].item(),
            "raw_value_after_complete_input": post[0, -1].item(),
        }


def load_value_model(model_dir, *, device="cuda", attention_backend="eager", weights_path=None):
    """Load every critic tensor, including its trained backbone and scalar head.

    FP32 storage is retained in the release. Inference uses BF16 backbone
    parameters and the original FP32 scalar head, matching native compute.
    """
    model_dir = Path(model_dir)
    config = GMPTConfig(**json.loads((model_dir / "backbone_config.json").read_text()))
    if attention_backend.startswith("flash_attention") and torch.device(device).type != "cuda":
        raise ValueError("FlashAttention requires a supported CUDA GPU.")
    configure_attention(attention_backend)
    config._attn_implementation = attention_backend
    weights_path = Path(weights_path) if weights_path is not None else model_dir / "model.safetensors"
    state = load_file(str(weights_path), device="cpu")
    head = state.get("value_head.weight")
    if head is None or tuple(head.shape) != (1, config.hidden_size):
        raise ValueError(
            "These weights are not this value model: expected a "
            f"value_head.weight of shape (1, {config.hidden_size})."
        )
    with torch.device("meta"):
        model = ValueModel(config)
    model.load_state_dict(state, strict=True, assign=True)
    del state
    model.backbone.to(device=device, dtype=torch.bfloat16)
    model.value_head.to(device=device, dtype=torch.float32)
    return model.eval().requires_grad_(False)


_TOKENIZER_CACHE = {}


def _load_tokenizers(model_dir):
    """Load and cache the tokenizer pair; repeated calls must not reparse it."""
    from transformers import AutoTokenizer
    from tokenizers import Tokenizer

    key = str(Path(model_dir).resolve())
    if key not in _TOKENIZER_CACHE:
        _TOKENIZER_CACHE[key] = (
            AutoTokenizer.from_pretrained(model_dir, local_files_only=True),
            Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json")),
        )
    return _TOKENIZER_CACHE[key]


def tokenize_pair(model_dir, prompt, response):
    """Use the shipped template; append the continuation without synthetic EOS."""
    tokenizer, raw = _load_tokenizers(model_dir)
    messages = [{"role": "user", "content": prompt}]
    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True,
    )["input_ids"]
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if prompt_ids != raw.encode(formatted, add_special_tokens=False).ids:
        raise ValueError("Tokenizer IDs differ from the shipped tokenizer.json; do not silently migrate it.")
    if response_ids != raw.encode(response, add_special_tokens=False).ids:
        raise ValueError("Continuation token IDs differ from tokenizer.json.")
    return prompt_ids, response_ids
