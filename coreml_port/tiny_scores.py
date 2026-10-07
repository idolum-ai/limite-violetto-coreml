"""Opt-in CPU score arithmetic for prefixes shorter than sixteen tokens.

The source CPU's three-token score GEMM matched a sequential FP32 sum across
128 BF16-exact products in the captured layer-38 failure. Core ML preserved
that explicit order in isolated and four-layer tests. For a short prefix only
the first sixteen key slots can be unmasked; the remaining score slots are
zeros that the caller's causal mask will discard. The ordinary raw-layout
einsum remains selected when prepared query rows are sixteen or more.

This is a bounded experimental policy, not a source-GEMM identity theorem for
all shapes. The caller must apply the usual key mask after calling ``scores``.
"""
import torch
from coreml_port.dynamic_queries import scores as ordinary_scores


def scores(query, expanded_keys):
    short = expanded_keys[:, :16]
    total = query[..., 0].unsqueeze(2) * short[..., 0].unsqueeze(1)
    for i in range(1, 128):
        total = total + query[..., i].unsqueeze(2) * short[..., i].unsqueeze(1)
    active = total.permute(0, 3, 1, 2) * .1
    inactive = torch.zeros_like(active[..., :1]).expand(-1, -1, -1, 2032)
    short_scores = torch.cat((active, inactive), dim=-1)
    full_scores = ordinary_scores(query, expanded_keys)
    rows = torch._shape_as_tensor(query)[1]
    return torch.where(rows < 16, short_scores, full_scores)
