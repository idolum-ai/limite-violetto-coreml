"""Experimental finite FP32 ties-even round with signed division.

The quotient round is designed to preserve signed zero in Core ML. The full
quantizer's final scale multiplication is still independently qualified and
must not be assumed to preserve Torch's negative-zero bits.
"""
from __future__ import annotations

import torch


def round_signed_even_v3(z: torch.Tensor) -> torch.Tensor:
    magnitude = torch.abs(z)
    low = torch.floor(magnitude)
    fraction = magnitude - low
    odd = low - 2.0 * torch.floor(low / 2.0)
    advance = torch.where(fraction == 0.5, odd, (fraction > 0.5).to(z.dtype))
    rounded_magnitude = low + advance
    ones = torch.ones_like(z)
    sign_denominator = torch.where(z < 0.0, -ones, ones)
    signed = rounded_magnitude / sign_denominator
    return torch.where(z == 0.0, z, signed)


def Q_signed_even_v3(x: torch.Tensor) -> torch.Tensor:
    exponent = torch.floor(torch.log2(torch.clamp(torch.abs(x), min=1e-30))) - 7
    powers = 2. ** torch.arange(-128, 128, dtype=torch.float32)
    step = powers[(exponent + 128).long()]
    return round_signed_even_v3(x / step) * step


def mil_round_signed_even_v3(mb, z):
    """MIL Builder expression for a finite FP32 quotient."""
    import numpy as np

    zero = np.float32(0.0)
    one = np.float32(1.0)
    negative_one = np.float32(-1.0)
    half = np.float32(0.5)
    two = np.float32(2.0)
    magnitude = mb.abs(x=z)
    low = mb.floor(x=magnitude)
    fraction = mb.sub(x=magnitude, y=low)
    odd = mb.sub(x=low, y=mb.mul(x=two, y=mb.floor(x=mb.real_div(x=low, y=two))))
    advance = mb.select(cond=mb.equal(x=fraction, y=half), a=odd,
                        b=mb.cast(x=mb.greater(x=fraction, y=half), dtype="fp32"))
    rounded_magnitude = mb.add(x=low, y=advance)
    negative_ones = mb.fill_like(ref_tensor=z, value=negative_one)
    positive_ones = mb.fill_like(ref_tensor=z, value=one)
    sign_denominator = mb.select(cond=mb.less(x=z, y=zero), a=negative_ones, b=positive_ones)
    signed = mb.real_div(x=rounded_magnitude, y=sign_denominator)
    return mb.select(cond=mb.equal(x=z, y=zero), a=z, b=signed)
