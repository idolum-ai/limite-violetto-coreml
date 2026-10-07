# Boundary-safe reciprocal correction

The separate one-site correction passes the controlled group0 experiment. It reproduces the original GPU-policy reciprocal error, corrects that value against an exact rational reference, and removes all 19 warm normalized-output differences. CPU outputs and all eight cache states stay byte-for-byte unchanged. This completes the bounded 06f milestone; GPU full-model qualification remains open.

## Results and gates

| Gate | Result | Evidence |
|---|---|---|
| C1 arithmetic design | Exact midpoint comparison retains the low-order information lost by the rejected helper | [Design](candidate-design.md), [math review](math-review.md) |
| C2 host correctness | PASS: 31,147 corrections and 249,176 exact identity checks, including the three previous failing boundaries | [Host result](host-1/results.json) |
| C3 standalone reproduction | FAIL: standalone division returns exact values and does not reproduce the saved GPU-policy seeds | [Native result](native-1/results.json) |
| C3b supplied-seed arithmetic | PASS: all 28 correction calls match the oracle; nearest CPU controls preserved | [Prospective design](native-design.md), [native result](native-1/results.json) |
| C4 in-model correction | PASS: 12 native calls, original seed reproduction, oracle-exact reciprocal and all 12 warm normalization stages exact | [Build](group0-1/results.json), [replay](group-replay-1/results.json), [independent summary](group-summary.json) |
| C5 remaining drift/full model | OPEN: warm downstream drift and earlier cold attention-finish discrepancy remain | [Next experiment](next-experiment.json) |
| C6 audit/preservation | Independent audits and both qualified CPU cohorts PASS; final checksum outcome recorded separately | [Review](review.md), [CPU preservation](cpu-preservation.json), [seal result](checksum-verification.json) |

The failed standalone reproduction remains failed. Its 36-call experiment includes 28 supplied-seed correction calls under a separately frozen gate. The subsequent in-model experiment independently reproduces the erroneous seed and proves the correction works at the actual target site; the standalone result alone did not justify that conclusion.

## What changed

The baseline is the prior round-only group0 candidate. At layer 0 post-attention reciprocal `var_5675`, the original division is retained as `reciprocal_seed`. The exact 144 operations exported by the native correction component are grafted after it, with six domain-guard operations. All other 46,034 baseline operations remain byte-identical; weights, state interfaces and model interfaces are preserved. A tapped clone adds outputs only, and its endpoints and all cache states match the plain candidate under both policies.

The helper compares exact midpoint boundaries and selects a neighboring FP32 value. Its contract is a positive finite denominator in `[0.5, 8]` and a positive normal seed no farther than one adjacent float from the exact nearest-even reciprocal. It uses no runtime FP64. Outside the denominator range, the group candidate returns the original seed. The recorded cases validate this bounded contract; they do not establish a general reciprocal implementation.

## Measured downstream effect

Both comparisons use the same saved CPU inputs and restored cache: cold32 and warm12. Relative to the prior round-only GPU-policy candidate:

| Metric | Before | After |
|---|---:|---:|
| Warm group output RMSE against CPU | 19.293515 | 6.705696 |
| Warm group output relative RMSE | 0.00210181 | 0.00073051 |
| Warm group output maximum absolute difference | 1,024 | 128 |
| Warm differing cache elements | 2,217 | 379 |
| Warm differing cache arrays | 6 | 3 |

Warm group output RMSE falls **65.24%**. The first differing warm cache moves from layer 1 K to layer 2 K. Cold output and cache arrays remain byte-identical to the prior GPU-policy candidate, so their errors are unchanged. The remaining warm output is also inexact. These are local numerical measurements, not whole-model accuracy, speed or completion percentages.

## Next step and limits

The [saved-graph inventory](remaining-reciprocals.md) finds 20 provable reciprocal-one sites, of which 19 remain unmodified. The next narrow probe is layer 1 pre-attention normalization `var_7154` (original operation 12034). First capture its denominator, seed and normalized values with endpoint/state controls; only apply the helper if an observed error satisfies its contract. Graph order and cache progression nominate this probe but do not establish causality. A cohort rewrite would currently mix uncharacterized normalization and softmax denominators.

The earlier cold attention-finish discrepancy remains a separate open question. Full-model GPU screening, partition/boundary/generation gates, performance measurements and actual hardware placement remain unqualified. `CPU_AND_GPU` names a permitted policy, not measured dispatch. Both original and promoted CPU models, runtime selection and numerical tolerances are unchanged.

[Outcome and evidence hashes](outcome.json), [next gates and evidence slots](next-experiment.json), [commands](commands.md), [current gate record](progress.json), [original frozen plan](protocol.json), [handoff checksums](SHA256SUMS).
