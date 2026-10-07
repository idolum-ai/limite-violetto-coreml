# Signed-zero diagnosis and proposed sign-factor replacement

The first signed-even operator gate fails and the group build remains gated. Its saved arrays establish a signed-zero failure, without identifying a native kernel or device-placement cause. Frozen operator/helper/builder files are unchanged.

## Observations from the completed operator

All 14 failing case/policy pairs differ only by expected `0x80000000` (-0) becoming actual `0x00000000` (+0). This was checked against every differing array element, not only the report's truncated examples. CPU_ONLY and CPU_AND_GPU have the same mismatch counts:

| Case (each policy) | Mismatches | Literal negative-zero inputs | Negative nonzero inputs rounding to zero |
| --- | ---: | ---: | ---: |
| directed-q3 | 15 | 3 | 12 |
| directed-q12 | 60 | 12 | 48 |
| directed-q32 | 160 | 32 | 128 |
| captured-c0-cpu-x_29 | 1,965 | 1,965 | 0 |
| captured-c0-gpu-x_29 | 1,965 | 1,965 | 0 |
| captured-c1-cpu-x_29 | 761 | 761 | 0 |
| captured-c1-gpu-x_29 | 761 | 761 | 0 |

Directed q1 passes both policies. Across both policies there are 11,374 differing elements. Thus a scalar/small-shape versus larger-shape execution difference is plausible, but neither its implementation nor actual CPU/GPU dispatch is established by this observation.

The existing compiled `artifacts/06d-gpu-rounding/operator-1/signed-even.mlmodelc/model.mil` retains:

```text
var_56 = mul(rounded_magnitude, -1)
signed = select(z < 0, var_56, rounded_magnitude)
var_60 = select(z == 0, z, signed)
y = mul(var_60, step)
```

No weighted-add lowering of these selects appears in this exported MIL. Such a lowering is one possible explanation, not an observed fact. The exported graph also cannot rule out later compiler fusion, sign canonicalization, or a preceding division's behavior. Only final outputs were exposed by operator1.

## Proposed sign transfer (root's formula)

Keep the same explicit nonnegative nearest-even magnitude `roundmag`. Replace both sign/zero selects with:

```text
magnitude = abs(z)
denominator = select(magnitude == 0, 1.0, magnitude)
signfactor = z / denominator
rounded = roundmag * signfactor
```

Under ordinary IEEE FP32 semantics for the intended bounded finite quotient, denominator is strictly positive. For nonzero z, division of z by its own magnitude gives exactly +1 or -1. For literal ±0, division by +1 retains ±0. Since roundmag is nonnegative, final multiplication gives the required sign even when the rounded magnitude is zero. The select handles only nonnegative magnitudes and +1; it never selects a negative zero.

For Q's clamp branch, smallest nonzero input `2^-149` divided by step `2^-107` yields quotient `2^-42`, so a nonzero quotient is normal and far from underflow. The denominator also avoids division by zero, NaN, or infinite reciprocals within this quotient domain. This is a semantic design argument, not proof of a particular native backend's lowering or division precision; the operator gate remains necessary.

A no-load symbolic Core ML conversion of this expression succeeded with rank `[1, unknown, 256]`. The exported graph preserved `denominator=select`, `signfactor=real_div(z,denominator)`, and `rounded=mul(roundmag,signfactor)`. Its inventory was 23 operations: 7 scalar constants and 16 nonconstant operations (`abs:1,floor:2,sub:2,real_div:2,mul:2,equal:2,greater:1,cast:1,select:2,add:1`). No package was saved or natively compiled/loaded/predicted during this check.

The bounded native zero-path probe should expose z, roundmag, denominator, signfactor, and final rounded result, with a raw-input identity and the old sign-select path as controls. Include literal ±0 and negative/positive sub-half, half, and just-above-half quotients; use q1/3/12/32 under both policies. Exact bit comparison is required. Output instrumentation can affect compilation, so a passing diagnostic must be followed by the output-only full operator gate before any group build.

## Rejected shortcuts

`sign(z)` cannot distinguish -0 from +0. Unguarded `z/abs(z)` produces 0/0. Reciprocal-sign extraction unnecessarily introduces infinities. Native-round correction depends on backend tie behavior and can create +0 by exact cancellation: `-1 - (-1) = +0` under nearest rounding. Similarly `r - (r - signed_round)` does not reliably preserve negative zero. The positive-denominator sign transfer avoids these specific problems while keeping native validation explicit.
