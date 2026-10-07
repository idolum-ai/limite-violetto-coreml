"""Experimental query geometry and score arithmetic for fixed 2048-key attention.

These expressions are copied from the tiny3 Core ML group-8 diagnostic in
scripts/gpu_dynamic_raw_attention.py. The tested policy puts real query rows
first and appends zeros to min(32, last absolute position + 1) rows. The weighted value product sees
only the actual query rows, sliced before multiplication. Both choices have
bounded Core ML CPU evidence; neither is a general arithmetic equivalence claim.
"""
import torch


def prepare(q, positions):
    rows = torch.minimum(torch.tensor(32, dtype=torch.int32), positions[-1] + 1)
    query = torch.cat((q, torch.zeros(1, 31, 10, 128)), dim=1)[:, :rows]
    pos = torch.cat((positions, torch.zeros(31, dtype=torch.int32)))[:rows]
    return query, pos


def scores(q, expanded_keys):
    return torch.einsum("bthd,bshd->bhts", q, expanded_keys) * .1


def product(weights, expanded_values, count):
    return torch.matmul(weights[:, :, :count], expanded_values).permute(0, 2, 1, 3)
