# GPU rounding correction and residual drift

The separate group0 candidate corrects the first key-cache mismatch while preserving the original CPU results. It reduces group-output RMSE by 88.99% on the first 32-token chunk and 49.19% on the following 12-token chunk. GPU-policy group outputs still differ; full-model GPU qualification remains open. This bounded correction experiment is complete; two narrower residual targets remain.

These are restored-state comparisons on the same saved CPU inputs and CPU cache, from one 44-token prompt. The percentages describe group0 output error, not whole-model accuracy, speed or completion progress. CPU_AND_GPU denotes a compute policy; actual hardware placement remains unverified.

## Gates and evidence

| Gate | Outcome | Evidence |
|---|---|---|
| R1: complete quantizer byte-exact to Torch, including zero signs | FAIL, also in unchanged native baseline; retained as failed | [v1](operator-1/results.json), [v2](operator-2/results.json), [native baseline](operator-baseline-1/results.json) |
| R1b: changed rounding operation byte-exact to Torch, full quantizer byte-exact to original native CPU | PASS, 48 guarded calls | [Prospective refinement](gate-refinement.md), [protocol](round-replacement-protocol.json), [raw result](round-replacement-1/results.json) |
| R2 structure: replace only 131 rounds | PASS; other 42,367 operations, weights and state interfaces unchanged | [Build](group0-1/results.json) |
| R2 CPU preservation | PASS; both endpoints and all eight states byte-exact in both chunks | [Six-call replay](group0-replay-1/results.json) |
| R2 GPU local exactness | FAIL; K0 repaired and V0 still exact, later states/output differ | [Independent recomputation](group-summary.json) |
| Residual instrumentation | Seven-tap load failure preserved; four- and six-tap controls PASS | [Failed replay](residual-replay-1/results.json), [four-tap controls](residual-replay-2/results.json), [six-tap controls](residual-replay-3/results.json) |
| R3 full-model expansion/qualification | NOT RUN; gated on local diagnosis | [Original frozen GPU screen](../06b-full-gpu/REPORT.md) |
| R4 measured device placement/performance | UNVERIFIED; no new claim | [Earlier empty trace results](../06c-gpu-localization/placement-summary.md) |
| Qualified CPU artifact/source preservation | PASS; original and promoted source/cache cohorts unchanged | [Preservation](cpu-preservation.json) |

## Rounding and signed-zero diagnosis

Both initial helpers complete 40 native calls normally, but 14 cases differ from Torch solely because negative zeros become positive zero. The original native quantizer produces byte-identical outputs to both helpers across all 40 arrays. Thus the strict full-quantizer failure is pre-existing native multiplication behavior, not a new candidate regression. Q1 preserves five negative zeros per policy; tested Q3/Q12/Q32 and captured sine-product cases lose signs. This does not establish behavior for every possible length.

The first zero-path diagnostic fails during conversion with zero predictions because clone elimination creates an input/output naming alias. The fresh [zero-path-2](zero-path-2/results.json) completes all 40 predictions and passes its old-endpoint control; the first zero-sign loss occurs at multiplication. All failed attempts remain intact.

Before v3 ran, [R1b](gate-refinement.md) separated exactness of the replaced round operation from preservation of the existing native quantizer. The v3 helper uses magnitude, floor and parity for halfway decisions, division by a tensor-valued sign to preserve negative zero, and explicit literal-zero passthrough. The exponent, scale and outer multiply remain unchanged. Direct rounding matches Torch bytes under both policies, including zeros; full Q matches original native CPU bytes. The strict full-Q Torch flag remains false, so the combined raw result's top-level `passed` is also false. The separate `bounded_replacement_gate_passed` is true. No full-model tolerance changed.

## Uninstrumented group0 result

The native candidate replaces every group0 `round` with the validated formula. Candidate CPU and original CPU produce identical endpoint bytes and all eight cache states for both saved chunks. Candidate GPU-policy K0 is now exact, correcting all 343 cold and 150 warm first-key-cache differences; V0 was already exact and stays so. K1 is the first remaining changed active cache.

| Shared-input/cache chunk | Original GPU-policy y RMSE vs CPU | Candidate RMSE | Reduction |
|---|---:|---:|---:|
| 32 tokens at position 0 | 95.990055 | 10.570169 | 88.99% |
| 12 tokens at position 32 | 37.975470 | 19.293515 | 49.19% |

The candidate y relative RMSE is 0.000153651 cold and 0.002101811 warm; max-absolute differences remain 256 and 1,024. These group hidden-value scales are not final-score errors. No full-model pass can be inferred from them. [Raw arrays and recomputation](group-summary.json).

## Residual probe and load failure

The first output-only clone exposes seven existing tensors from rotary query through attention and layer1 key projection. Its build preserves all 46,035 candidate operations and compiles successfully. During replay, the GPU-policy tapped model fails loading with Core ML error -14: data-dependent shapes are disabled for `scores_1`, `weights_1`, and `x_37`. Six calls finish before that failure; all eight planned controls do not complete, so no tap values from that attempt are interpreted. This is an instrumentation load restriction, not a new numerical crash.

The fresh four-tap clone removes those three exports and retains `q_3`, `value_19`, `x_93`, and `x_97`. It preserves every operation and changes only output lists. [Build](residual-taps-2/results.json), [frozen replay protocol](residual-replay-2-protocol.json).

## Reproduction and review

[Commands](commands.md), [independent review](review.md), [original protocol](protocol.json), [operator trail](../../OPERATOR-TRAIL.md). All artifacts are separate from the qualified CPU selections. Failed or superseded code, protocols, models, logs and raw arrays are retained rather than overwritten.

The reduced replay completes normally with all eight calls and all four policy/chunk controls byte-exact. `q_3` is exact under both policies in both chunks. `value_19` is already different: the first mismatch is cold `[0,19,3]` (CPU24.375, GPU-policy24.5) and warm `[0,8,25]` (CPU0.6015625, GPU-policy0.59765625). The next layer normalized input and raw key also differ. This brackets the residual after exact Q/K/V and by the first layer final hidden; it does not yet identify an attention or MLP kernel. [Validated four-tap replay](residual-replay-2/results.json).


## Final residual boundary and next milestone

The six-tap replay completes all eight calls with byte-exact same-policy endpoint/all-state controls. Overlapping query and final-hidden checkpoints agree with the four-tap experiment.

| Chunk | Query `q_3` | Attention residual `value_17` | MLP input normalization `x_69` | Later MLP / final hidden |
|---|---|---|---|---|
| Cold, 32 tokens | Exact | First observed difference: `[0,19,3]`, CPU4.46875 vs GPU-policy4.5 | Different input; output also differs | Differ |
| Warm, 12 tokens | Exact | Exact | First observed difference: `[0,8,23]`, CPU-0.87890625 vs GPU-policy-0.8828125 | Differ |

The warm case isolates an equal-input normalization discrepancy: divergence lies between the exact attention residual and its normalized/quantized output. The cold case has an earlier unresolved attention-finish discrepancy. These are sampled boundaries, not proof of the specific responsible operation or that one correction will solve both.

Next, capture the warm normalization's square, reduction, scale and pre/post-quantization values with the same endpoint/state controls. Use the first equal-input differing operation to define a separate arithmetic experiment. Then return to the cold attention-finish interval. Full-model rebuilding remains deferred until those local residuals are understood. [Next experiment and evidence slots](next-experiment.json).

The original and promoted CPU source/compiled identities and pinned sources pass preservation checks. No runtime selection, original acceptance threshold, CPU promotion or qualified artifact was changed. [Final gate record](outcome.json), [independent residual recomputation](residual-summary-3.json), [whole-project handoff verification](checksum-verification.json).

The warm normalization checkpoint map is recorded in the next experiment: square → ordered sum → divide by1280 → add2^-23 → sqrt → reciprocal → multiply → quantization step/quotient/round. It is a structural plan, not evidence that any particular stage is responsible.
