# Native reciprocal controls and external-seed component gate

Freeze before native execution. Keep original C3 saved-baseline reproduction unchanged. A standalone reciprocal may choose a different plan from the full group, so nonreproduction remains a failed reproduction gate and forbids an in-model repair claim.

Add a separate prospective C3b arithmetic gate: correction is a function of an explicit denominator and approximate reciprocal seed. Feed the saved GPU reciprocal directly, as well as exact nearest and adjacent FP32 seeds, under both policies. This tests the component even if the isolated division does not reproduce the full group's discrepancy. Passing C3b does not override C3 or establish group fidelity. A fresh in-model experiment remains mandatory.

- Reuse all four previously frozen denominator cases (108 elements), including values immediately below1/2/4.
- For every case, test nearest reciprocal plus immediate lower/upper seeds; also test the two saved GPU-seed cases. Fourteen correction cases per policy.
- Independently run the original `ones_like(d)/d` control on four denominator cases per policy.
- Total:36 guarded native predictions, two models, two policies.
- Exact rational nearest-FP32 reference; all output bytes must match for component qualification. CPU nearest-seed outputs must preserve original CPU division bytes. No tolerance changes.
- Build both native models with the direct MIL helper and EMPTY conversion. Retain `real_div` in the control. The correction graph contains no division/inverse. Record operation/package/compiled hashes and input/output arrays.
- Full group source, weights, runtime and acceptance remain frozen. No speed or measured-device claim.

The group builder will graft the exact exported correction operations from a passing C3b native component, rather than reconvert a separate expression. Input names are d/seed and the final output is value; all intermediate types and operation bytes are bound in the component record.
