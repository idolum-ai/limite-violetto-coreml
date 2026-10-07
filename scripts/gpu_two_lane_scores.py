"""Experimental two-lane short-prefix scores for bounded Core ML probes.

This is deliberately not in coreml_port: the two-lane reduction does not
reproduce the pinned CPU GEMM's raw FP32 scores on independent test vectors.
"""
import torch
from coreml_port.dynamic_queries import scores as ordinary_scores


def scores(query, expanded_keys):
    short = expanded_keys[:, :16]
    first = query[..., 0].unsqueeze(2) * short[..., 0].unsqueeze(1)
    second = query[..., 1].unsqueeze(2) * short[..., 1].unsqueeze(1)
    for i in range(2, 128):
        term = query[..., i].unsqueeze(2) * short[..., i].unsqueeze(1)
        if i % 2:
            second = second + term
        else:
            first = first + term
    active = (first + second).permute(0, 3, 1, 2) * .1
    tail = torch.zeros_like(active[..., :1]).expand(-1, -1, -1, 2032)
    short_scores = torch.cat((active, tail), dim=-1)
    normal = ordinary_scores(query, expanded_keys)
    rows = torch._shape_as_tensor(query)[1]
    return torch.where(rows < 16, short_scores, normal)
