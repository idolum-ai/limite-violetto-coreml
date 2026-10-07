"""Experimental pinned Torch CPU arithmetic for 128-wide XSA normalization."""
import torch

from coreml_port.ordered_norm import ordered_sum


def ordered_l2(value):
    """Sum squares in four sequential lanes, fold left, and take square root."""
    width = value.shape[-1]
    if width != 128:
        raise ValueError("Only the model's 128-wide XSA vectors are supported")
    lanes = value[..., :4] * value[..., :4]
    for j in range(4, width, 4):
        q = value[..., j:j + 4]
        lanes = lanes + q * q
    squared = ((lanes[..., 0:1] + lanes[..., 1:2]) + lanes[..., 2:3]) + lanes[..., 3:4]
    return torch.sqrt(squared)


def ordered_xsa_normalize(value):
    return value / ordered_l2(value).clamp_min(1e-4)


def ordered_xsa_dot(y, vn):
    return ordered_sum(y * vn)
