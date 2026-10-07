"""Empirical four-lane short-prefix score variants for Torch-only screening."""
import torch
from coreml_port.dynamic_queries import scores as ordinary_scores


def scores(query, expanded_keys, fold="adjacent"):
    short = expanded_keys[:, :16]
    lanes = [query[..., i].unsqueeze(2) * short[..., i].unsqueeze(1)
             for i in range(4)]
    for i in range(4, 128):
        term = query[..., i].unsqueeze(2) * short[..., i].unsqueeze(1)
        lanes[i % 4] = lanes[i % 4] + term
    if fold == "adjacent":
        total = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
    elif fold == "forward":
        total = ((lanes[0] + lanes[1]) + lanes[2]) + lanes[3]
    else:
        raise ValueError(fold)
    active = total.permute(0, 3, 1, 2) * .1
    tail = torch.zeros_like(active[..., :1]).expand(-1, -1, -1, 2032)
    short_scores = torch.cat((active, tail), dim=-1)
    normal = ordinary_scores(query, expanded_keys)
    rows = torch._shape_as_tensor(query)[1]
    return torch.where(rows < 16, short_scores, normal)
