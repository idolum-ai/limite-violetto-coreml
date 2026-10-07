"""Experimental FP32 reciprocal correction by exact midpoint decisions.

Contract: d is finite FP32 in [0.5, 8], r0 is a positive normal FP32 seed at
most one representable neighbor from RN(1/d), and shapes match. Arithmetic
requires ordinary FP32 operations without unsafe reassociation. Host proof
checks do not establish a native backend's behavior. No runtime FP64 is used.
"""
from __future__ import annotations


def reciprocal_diagnostics(d, r0):
    """Torch diagnostic intermediates; final output is an existing FP32 neighbor."""
    import torch
    both, either = torch.logical_and, torch.logical_or

    def two_sum(a, b):
        total = a + b
        bb = total - a
        error = (a - (total - bb)) + (b - bb)
        return total, error

    # Error-free product (Dekker splitting); all products are safely normal
    # or exact zero on the declared domain and seed contract.
    ta = d * 4097.0
    dh = ta - (ta - d)
    dl = d - dh
    tb = r0 * 4097.0
    rh = tb - (tb - r0)
    rl = r0 - rh
    product = d * r0
    e1 = product - dh * rh
    e2 = e1 - dl * rh
    e3 = e2 - dh * rl
    error = dl * rl - e3
    residual_hi = 1.0 - product

    # Include the neighbors just below 1/8 and above 2 at the domain edges.
    up = torch.full_like(r0, 2.0 ** -27)
    scale = torch.full_like(r0, 2.0 ** 27)
    at_power = r0 == 2.0 ** -4
    for exponent in (-3, -2, -1, 0, 1):
        boundary = 2.0 ** exponent
        up = torch.where(r0 >= boundary, torch.full_like(r0, 2.0 ** (exponent - 23)), up)
        scale = torch.where(r0 >= boundary, torch.full_like(r0, 2.0 ** (23 - exponent)), scale)
        at_power = either(at_power, r0 == boundary)
    down = torch.where(at_power, up * 0.5, up)
    scaled = r0 * scale
    odd = scaled - 2.0 * torch.floor(scaled * 0.5)
    even = odd == 0.0
    upper_term = d * (up * 0.5)
    lower_term = d * (down * 0.5)
    upper_hi, upper_lo = two_sum(error, upper_term)
    lower_hi, lower_lo = two_sum(error, -lower_term)

    # residual_hi is exact by Sterbenz. Each threshold is hi+lo, with hi
    # correctly rounded and lo its exact remainder. Never round hi+lo again.
    upper_tie = both(residual_hi == upper_hi, upper_lo == 0.0)
    lower_tie = both(residual_hi == lower_hi, lower_lo == 0.0)
    not_even = torch.logical_not(even)
    go_up = either(either(residual_hi > upper_hi, both(residual_hi == upper_hi, upper_lo < 0.0)), both(upper_tie, not_even))
    go_down = either(either(residual_hi < lower_hi, both(residual_hi == lower_hi, lower_lo > 0.0)), both(lower_tie, not_even))
    previous = r0 - down
    following = r0 + up
    corrected = torch.where(go_up, following, torch.where(go_down, previous, r0))
    return dict(corrected=corrected, product=product, product_error=error,
                residual_hi=residual_hi, up=up, down=down, scaled=scaled, even=even,
                upper_term=upper_term, lower_term=lower_term,
                upper_hi=upper_hi, upper_lo=upper_lo, lower_hi=lower_hi, lower_lo=lower_lo,
                previous=previous, following=following, go_up=go_up, go_down=go_down)


def correct_reciprocal(d, r0):
    """Correct a supplied near-neighbor seed; never computes a replacement seed."""
    return reciprocal_diagnostics(d, r0)['corrected']


def mil_reciprocal_diagnostics(mb, d, r0):
    """Matching MIL Builder expression; caller controls model/output names."""
    import numpy as np
    c = np.float32
    add = lambda a, b: mb.add(x=a, y=b)
    sub = lambda a, b: mb.sub(x=a, y=b)
    mul = lambda a, b: mb.mul(x=a, y=b)
    eq = lambda a, b: mb.equal(x=a, y=b)
    less = lambda a, b: mb.less(x=a, y=b)
    greater = lambda a, b: mb.greater(x=a, y=b)
    both = lambda a, b: mb.logical_and(x=a, y=b)
    either = lambda a, b: mb.logical_or(x=a, y=b)
    select = lambda cond, a, b: mb.select(cond=cond, a=a, b=b)
    fill = lambda value: mb.fill_like(ref_tensor=r0, value=c(value))

    def two_sum(a, b):
        total = add(a, b)
        bb = sub(total, a)
        return total, add(sub(a, sub(total, bb)), sub(b, bb))

    ta = mul(d, c(4097))
    dh = sub(ta, sub(ta, d))
    dl = sub(d, dh)
    tb = mul(r0, c(4097))
    rh = sub(tb, sub(tb, r0))
    rl = sub(r0, rh)
    product = mul(d, r0)
    e1 = sub(product, mul(dh, rh))
    e2 = sub(e1, mul(dl, rh))
    e3 = sub(e2, mul(dh, rl))
    error = sub(mul(dl, rl), e3)
    residual_hi = sub(c(1), product)
    up, scale = fill(2.0 ** -27), fill(2.0 ** 27)
    at_power = eq(r0, c(2.0 ** -4))
    for exponent in (-3, -2, -1, 0, 1):
        boundary = c(2.0 ** exponent)
        predicate = mb.greater_equal(x=r0, y=boundary)
        up = select(predicate, fill(2.0 ** (exponent - 23)), up)
        scale = select(predicate, fill(2.0 ** (23 - exponent)), scale)
        at_power = either(at_power, eq(r0, boundary))
    down = select(at_power, mul(up, c(.5)), up)
    scaled = mul(r0, scale)
    odd = sub(scaled, mul(c(2), mb.floor(x=mul(scaled, c(.5)))))
    even = eq(odd, c(0))
    upper_term = mul(d, mul(up, c(.5)))
    lower_term = mul(d, mul(down, c(.5)))
    upper_hi, upper_lo = two_sum(error, upper_term)
    lower_hi, lower_lo = two_sum(error, mul(lower_term, c(-1)))
    upper_tie = both(eq(residual_hi, upper_hi), eq(upper_lo, c(0)))
    lower_tie = both(eq(residual_hi, lower_hi), eq(lower_lo, c(0)))
    not_even = mb.logical_not(x=even)
    go_up = either(either(greater(residual_hi, upper_hi), both(eq(residual_hi, upper_hi), less(upper_lo, c(0)))), both(upper_tie, not_even))
    go_down = either(either(less(residual_hi, lower_hi), both(eq(residual_hi, lower_hi), greater(lower_lo, c(0)))), both(lower_tie, not_even))
    previous, following = sub(r0, down), add(r0, up)
    corrected = select(go_up, following, select(go_down, previous, r0))
    return dict(corrected=corrected, product=product, product_error=error,
                residual_hi=residual_hi, up=up, down=down, scaled=scaled, even=even,
                upper_term=upper_term, lower_term=lower_term,
                upper_hi=upper_hi, upper_lo=upper_lo, lower_hi=lower_hi, lower_lo=lower_lo,
                previous=previous, following=following, go_up=go_up, go_down=go_down)


def mil_correct_reciprocal(mb, d, r0):
    return mil_reciprocal_diagnostics(mb, d, r0)['corrected']
