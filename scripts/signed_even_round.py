"""Experimental finite-range signed ties-to-even rounding for Core ML probes.

This module does not replace the qualified quantizer. `Q_signed_even` preserves
the qualified exponent/step expression and changes only the quotient round.
The optional MIL helper is for isolated graph surgery experiments.
"""
from __future__ import annotations

import torch


def round_signed_even(z: torch.Tensor) -> torch.Tensor:
    """Round finite FP32 quotients to even, preserving negative rounded zero."""
    magnitude = torch.abs(z)
    low = torch.floor(magnitude)
    fraction = magnitude - low
    odd = low - 2.0 * torch.floor(low / 2.0)
    advance = torch.where(fraction == 0.5, odd, (fraction > 0.5).to(z.dtype))
    rounded_magnitude = low + advance
    signed = torch.where(z < 0.0, -rounded_magnitude, rounded_magnitude)
    return torch.where(z == 0.0, z, signed)


def Q_signed_even(x: torch.Tensor) -> torch.Tensor:
    """Match `graphs.Q` on bounded finite FP32 inputs with explicit tie parity."""
    exponent = torch.floor(torch.log2(torch.clamp(torch.abs(x), min=1e-30))) - 7
    powers = 2. ** torch.arange(-128, 128, dtype=torch.float32)
    step = powers[(exponent + 128).long()]
    return round_signed_even(x / step) * step


def mil_round_signed_even(mb, z):
    """MIL Builder expression for a finite FP32 quotient; returns an FP32 Var.

    The caller owns its surrounding step computation and output names. All
    scalar constants are FP32 to avoid accidental precision or type changes.
    """
    import numpy as np

    zero = np.float32(0.0)
    half = np.float32(0.5)
    two = np.float32(2.0)
    negative_one = np.float32(-1.0)
    magnitude = mb.abs(x=z)
    low = mb.floor(x=magnitude)
    fraction = mb.sub(x=magnitude, y=low)
    odd = mb.sub(x=low, y=mb.mul(x=two, y=mb.floor(x=mb.real_div(x=low, y=two))))
    advance = mb.select(cond=mb.equal(x=fraction, y=half), a=odd,
                        b=mb.cast(x=mb.greater(x=fraction, y=half), dtype="fp32"))
    rounded_magnitude = mb.add(x=low, y=advance)
    signed = mb.select(cond=mb.less(x=z, y=zero),
                       a=mb.mul(x=rounded_magnitude, y=negative_one),
                       b=rounded_magnitude)
    return mb.select(cond=mb.equal(x=z, y=zero), a=z, b=signed)
