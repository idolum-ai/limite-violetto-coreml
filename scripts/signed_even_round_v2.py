"""Experimental finite FP32 ties-even rounding with zero-sign factorization.

This is a separate candidate from signed_even_round.py. It is not qualified by
the earlier operator-1 result and must pass an isolated native operator gate.
"""
from __future__ import annotations

import torch


def round_signed_even_v2(z: torch.Tensor) -> torch.Tensor:
    """Avoid selecting a signed zero; multiply the rounded magnitude by sign."""
    magnitude = torch.abs(z)
    low = torch.floor(magnitude)
    fraction = magnitude - low
    odd = low - 2.0 * torch.floor(low / 2.0)
    advance = torch.where(fraction == 0.5, odd, (fraction > 0.5).to(z.dtype))
    rounded_magnitude = low + advance
    denominator = torch.where(magnitude == 0.0, torch.ones_like(magnitude), magnitude)
    sign_factor = z / denominator
    return rounded_magnitude * sign_factor


def Q_signed_even_v2(x: torch.Tensor) -> torch.Tensor:
    """Keep the sealed quantizer's exponent and scale; replace quotient round."""
    exponent = torch.floor(torch.log2(torch.clamp(torch.abs(x), min=1e-30))) - 7
    powers = 2. ** torch.arange(-128, 128, dtype=torch.float32)
    step = powers[(exponent + 128).long()]
    return round_signed_even_v2(x / step) * step


def mil_round_signed_even_v2(mb, z):
    """MIL Builder expression for a finite FP32 quotient."""
    import numpy as np

    zero = np.float32(0.0)
    one = np.float32(1.0)
    half = np.float32(0.5)
    two = np.float32(2.0)
    magnitude = mb.abs(x=z)
    low = mb.floor(x=magnitude)
    fraction = mb.sub(x=magnitude, y=low)
    odd = mb.sub(x=low, y=mb.mul(x=two, y=mb.floor(x=mb.real_div(x=low, y=two))))
    advance = mb.select(cond=mb.equal(x=fraction, y=half), a=odd,
                        b=mb.cast(x=mb.greater(x=fraction, y=half), dtype="fp32"))
    rounded_magnitude = mb.add(x=low, y=advance)
    denominator = mb.select(cond=mb.equal(x=magnitude, y=zero), a=one, b=magnitude)
    sign_factor = mb.real_div(x=z, y=denominator)
    return mb.mul(x=rounded_magnitude, y=sign_factor)
