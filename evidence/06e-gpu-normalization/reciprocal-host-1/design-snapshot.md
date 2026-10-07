# N4 bounded reciprocal refinement diagnostic

This standalone experiment tests one compensated Newton correction on saved positive FP32 denominators. It changes no group model, arithmetic helper, runtime, weight, or qualification gate. The observed warm reciprocal difference must reproduce in isolation before the experiment can support a correction claim; CPU_AND_GPU permission does not establish actual GPU execution.

Two models are built by the root runner with FP32 and an EMPTY conversion pipeline:

- Control: `r0 = ones_like(d) / d`, returning r0.
- Diagnostic: the same r0, `e = fma(-d,r0,1)`, `r1 = fma(r0,e,r0)`, returning r0, e, r1.

Here `fma` is the existing, read-only `coreml_port/source_exp.py` compensated multiply-add using FP32 splitting by4097. The entire helper file SHA and snapshot are pinned. This helper is empirically bounded, not a general IEEE fused-multiply-add guarantee. Separate ordinary multiplication may erase a Newton residual by rounding d*r0 to exactly1, which motivates compensation. Native compiler reassociation can still invalidate the intended operation order.

The converted graphs must retain exactly one `real_div` and no `inverse`; only const/fill_like/real_div/mul/sub/add/identity are allowed. The runner records the realized operation inventory/hash, division protobuf, and input descriptor, then saves package/compiled inventories. This check applies to conversion; later native lowering remains unverified.

Exactly four frozen input cases run under each policy and each model, for16 predictions:

1. Warm common denominator [1,12,1], with distinct saved CPU and GPU-permitted reciprocal expectations. Its denominator must be byte-identical between the two saved policy runs.
2. Cold CPU denominator [1,32,1], with saved CPU reciprocal expectation.
3. Cold GPU-permitted denominator [1,32,1], with its own saved reciprocal expectation.
4. Directed [1,32,1]: powers1/2/4 and immediate FP32 neighbors; min/max captured denominator and neighbors; all12 warm denominators; and1.25,1.5,1.75,2.5,3.

The safe input domain is frozen to0.5≤d≤8. Captured d is approximately1.53..2.95. In this range the reciprocal and4097 splitting products are finite and well within FP32 normal range; exact-zero error terms are allowed. The directed case checks limited positive finite behavior, not zero, negative, infinite, or general-range reciprocal semantics. Captured array archive hashes, metadata, and exact case byte hashes are frozen before launch. No random generation or adaptive search occurs.

The oracle rounds the exact rational1/d to FP32. It binary-searches positive finite FP32 bit patterns using integer-rational comparisons, compares exact distances to adjacent floats, and resolves midpoint ties by even significand parity. It uses no floating-point division. Static checks include exact powers and explicit rational midpoint parity cases. Host Torch execution checks the control and candidate against this oracle before native compilation.

All16 numerical calls are saved even if reproduction, reference accuracy, or CPU preservation fails. Runtime exceptions can prevent completion and are reported as failures. The independent gates are:

- Saved-control reproduction: isolated control equals the corresponding saved reciprocal for warm CPU, warm GPU-permitted, cold CPU/CPU, and cold GPU/GPU.
- Diagnostic baseline preservation: diagnostic r0 equals standalone control r0 on all8 policy/case pairs.
- Reference accuracy: diagnostic r1 equals the exact-rational FP32 oracle on all8 pairs.
- CPU preservation: CPU diagnostic r1 equals original CPU control r0 on all4 cases, and CPU control itself equals the oracle on all4 cases.
- All outputs finite, all inputs unchanged, schemas correct,16 plain predictions/16 GIL releases/zero state predictions, normal completion, and all pinned files and inventories preserved.

If either reproduction gate fails, the decision is `nonreproduction_no_correction_claim`, even when refined outputs match the oracle. The runner still records every numerical comparison. Passing every gate means only that this bounded standalone candidate may proceed to a separate group experiment. It does not promote a model, establish device placement, repair cold attention drift, or satisfy full-model qualification.

Existing prediction/cleanup guards install before conversion, compilation, loading, or prediction. The runner records runtime provenance and requires equality to the saved replay, saves source/helper/protocol/design/replay snapshots, releases model references between arms, and records cleanup/guard counters. No reference full model is loaded.

Root launch:

```sh
.venv/bin/python scripts/probe_reciprocal_refinement.py --attempt reciprocal-1 --protocol evidence/06e-gpu-normalization/reciprocal-protocol.json
```

The original milestone `protocol.json` is preserved. This separate reciprocal protocol is immutable for this attempt; fresh attempt paths are mandatory.

## Prelaunch result: failed host gate

The exact-rational and host-control checks passed, but this compensated correction failed CPU preservation on three directed denominators: the immediate FP32 predecessor of1,2,and4. It returned exactly1,0.5,and0.25 instead of the next representable float above those values. All three captured-input cases passed. The directed cases are retained and the gate is not relaxed.

`reciprocal-protocol.json` records `static_preflight.passed=false` and complete case comparisons. The runner will abort at the host preflight before conversion/compilation/prediction. No native model was built and no prediction was performed. The planned16-call native diagnostic is blocked for this candidate; a redesigned correction requires a separately reviewed attempt.

The three failing inputs were checked again using exact rational arithmetic. Both (a) the single exact Newton expression rounded once and (b) two correctly rounded FMA operations give the oracle result on these inputs. The first helper call's residual also matches correctly rounded exact1-d*r0. The second compensated helper call gives a different final bit pattern from correctly rounded r0+r0*e. Thus these particular failures are localized to the host compensated multiply-add realization of the final correction, not an established defect in the ideal Newton formula. This result does not prove one Newton iteration is universally sufficient. No compiler or GPU cause is claimed because no model conversion or native execution occurred.
