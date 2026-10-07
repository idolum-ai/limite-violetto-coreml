# V3 round-only replacement and boundary

V3 restores the nonnegative nearest-even magnitude with division by a tensor of ±1:

```text
negative_ones = fill_like(z, -1)
positive_ones = fill_like(z, +1)
denominator = select(z < 0, negative_ones, positive_ones)
signed = rounded_magnitude / denominator
result = select(z == 0, z, signed)
```

For nonzero finite z, exact division by ±1 gives the required sign with no magnitude error, including a negative rounded zero. Literal -0 has a false `z < 0` condition, so the final zero passthrough is necessary to retain its sign. This formula depends on the backend preserving the division and final selection semantics; symbolic correctness does not replace the native R1b gate.

The initial draft used scalar ±1 branches, which MIL Builder inferred as a scalar select result despite a tensor condition. Before native execution, the helper was corrected to tensor-shaped `fill_like` branches. The final builder rejects all nonconstant scalar template outputs; no scalar type override remains. It verifies exactly two fills, each referencing z, with scalar constant values -1 and +1 respectively. Prior v1/v2 builders and helpers are unchanged.

The final frozen helper is `scripts/signed_even_round_v3.py`, SHA-256 `d1e7af3cfbc0b493b1f481984bdbf0469cce693860c1bd0133d86b53a3a7bcb0`. The builder is `scripts/build_signed_even_group_v3.py`, SHA-256 `13ba011149121669465b61da5aa3f2a9bed2dc63657db99088367f6b6d1a87f1`, with arithmetic policy `explicit-signed-ties-even-v3-round-only` and explicit helper/builder source path fields.

The real helper's symbolic conversion retains variable `real_div` with select-produced denominator and final zero passthrough under both EMPTY and DEFAULT pipelines. The DEFAULT pipeline changes the separate constant divide-by-two used in parity into multiplication; it does not change the sign-restoration division. This check covers Core ML Tools conversion only, not later native compilation. See `v3-sign-division-mil-check.json`.

The builder's in-memory original-group0 preflight replaces all131 rounds, preserves42,367 other operations byte-for-byte, preserves all original output names/types and non-operation metadata, supports a single-site selection, and rejects unknown sites. See `group-builder-v3-preflight.json`. It has not created or compiled a package or run a native prediction.

The unchanged multiplication by the quantization step remains outside the replacement. Baseline-control findings require distinct numerical gates: isolate the actual replacement round against Torch byte-for-byte, then compare complete native Q against original native Q and require full group CPU outputs/state bytes to match the original native group. Earlier strict full-Q-versus-Torch failures remain failed; this builder does not redefine or relax them.

Future scope remains unimplemented. `original-14-component-round-inventory.json` is a read-only inventory of the original cohort, with every component's serialized model hash and operation counts. All14 components use CoreML8 and flat blocks; group6 and group11 have149 and150 round sites, while embedding/head each have one. No generalized or full-cohort builder has been authorized by this inventory.
