"""Experimental FP32 value multiplication in fixed pairs of query rows.

The fixed-capacity attention path has at most 32 queries and 2048 keys.
Pair-sized Core ML products reproduce the saved CPU capacity-case product
exactly where a 32-row product does not. Full-model qualification is separate.
"""
import torch


def product(weights, expanded_values, count):
    padded = torch.cat((weights, torch.zeros(1, 10, 31, 2048)), dim=2)[:, :, :32]
    pieces = [padded[:, :, i:i+2] @ expanded_values for i in range(0, 32, 2)]
    return torch.cat(pieces, dim=2).permute(0, 2, 1, 3)[:, :count]
