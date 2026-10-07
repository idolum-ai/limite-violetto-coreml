# Research roadmap

The next numerical experiment is a **separate, bounded scaled reciprocal correction** for the warm layer2 key-normalization error. It was planned, not implemented, when the archived investigation ended.

The [06g report](../evidence/06g-gpu-residual/REPORT.md) and [machine-readable next experiment](../evidence/06g-gpu-residual/next-experiment.json) are the detailed handoff. With identical projection and denominator, one quotient crosses from exactly −169.5 to −169.49998474121094; rounding changes a key from −170 to −169 before quantization. Saved-tensor substitution removes that local key difference. It does not prove that a native correction restores full-model GPU outputs.

1. Freeze a power-of-two wrapper for positive finite FP32 denominators in `[4,64]`: scale the denominator by 1/8, the seed by 8, apply the already bounded helper, and scale the corrected result by 1/8.
2. Verify scaling, neighbor mapping, directed power/guard boundaries, out-of-domain fallback, and exact CPU preservation with a rational nearest-even oracle.
3. Test the actual rank-4 native supplied-seed component under both compute policies.
4. Apply only that additional site in a separate group0 candidate. Require unchanged CPU endpoints and all eight states, then measure the key-cache and downstream residuals on the same fixtures.
5. Audit raw arrays and preserve both qualified CPU cohorts before considering any full-model expansion.

Cold drift, standalone seed reproduction, actual device placement, and full GPU numerical/lifecycle/partition/boundary/generation gates remain open. No GPU speedup claim follows from the local counterfactual. ANE work and original NVIDIA-runtime parity are also unqualified.

For practical inference, the portable CPU path remains preferable in the retained measurements. A separate future packaging effort could make clean-machine setup easier, but this import intentionally preserves the existing experiment structure and arithmetic.
