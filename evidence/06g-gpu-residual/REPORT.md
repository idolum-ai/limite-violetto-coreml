# Remaining GPU-policy normalization drift

Status: complete for bounded localization. The remaining warm layer2 key-cache mismatch is traced to a reciprocal in key normalization, `var_14324`. With identical projection and denominator, nine GPU-policy reciprocal seeds differ by one FP32 step; one resulting quotient crosses a rounding boundary. A controlled saved-tensor substitution removes the rounded-key difference. No new native correction is applied in this milestone.

## What the three probes established

All three output-only clones preserve the same 46,185 model operations, weights and original interfaces. Each runs eight guarded stateful calls on the same cold32/warm12 inputs with fresh restored CPU state. All **24 calls** finish normally; all **12 control cells** reproduce both the saved06f baseline and same-policy original endpoints/all eight cache states. Each independent analysis verifies 28 archives, with shared reference archives across probes.

| Probe | Warm result | Evidence |
|---|---|---|
| Layer1 pre-attention normalization | Reciprocal var_7154 differs at 1/12 positions, but the rounded output x_93 is exact | [Replay](replay-1/results.json), [analysis](analysis.json) |
| Attention/MLP to layer2 key interval | Attention, MLP, residual and layer2 rounded normalization x_193 are exact; first exposed surviving key difference is x_219 | [Replay](interval-replay-1/results.json), [analysis](interval-analysis.json) |
| Layer2 key projection/norm/rotary path | Projection and denom_21 are exact; reciprocal drift changes one normalized key, carried into rotary output and K2 cache | [Replay](key-replay-1/results.json), [analysis](key-analysis.json) |

The first two probes are useful negative findings. Var_7154's warm denominator range is 8.445766–16.031174, outside the existing helper domain. Layer2 pre-attention reciprocal var_13395 also differs at three positions, with denominators about111.85–1997.43, but its output x_193 is exact after rounding. Neither site is justified as a correction target for this captured warm trajectory. The layer1 post-attention raw normalization also differs transiently, but its rounded output is exact; its underlying operation is not isolated by these taps.

## The surviving key error

Warm raw key projection x_197, quantized projection value_41 and square-root denominator denom_21 match across policies. CPU reciprocal values are exact against the rational nearest-even oracle. GPU policy has nine errors among24 seeds: seven −1 ULP and two +1 ULP. Both policy seeds remain within one neighbor of the oracle.

At `[0,0,0,73]`, the normalization quotient changes from CPU **−169.5** to GPU-policy **−169.49998474121094**. Both signed rounding operations correctly follow ties-to-even: their different inputs produce −170 and −169. The quantized key k_49 therefore changes from **−0.33203125** to **−0.330078125**. That numeric difference survives the cosine arm, rotary sum x_219 and final key k_51, reaching cache position `[0,32,0,73]` in layer2 K. The sine arm is zero at that position. Separately, 775 warm sine-output bit differences are signed-zero-only and do not explain this numeric mismatch.

The key-normalization denominator ranges are24.060638–58.723000 warm and30.10–58.45 cold, beyond the frozen `[0.5,8]` correction domain. The original helper is not silently broadened or inserted here.

## Controlled counterfactual

The [frozen host experiment](key-counterfactual-design.md) reconstructs raw multiplication, quotient, nearest-even round and final multiplication from saved inputs and quantization steps. All four stages reproduce native saved arrays bit-for-bit in both policies and both chunks, including signed zeros. Forty original-versus-tapped endpoint/state comparisons are checked independently.

For the warm chunk, projection, denominator and quantization step are identical. Substituting **only the CPU reciprocal seed** into the saved GPU computation removes all962 raw/quotient differences and the single rounded-key difference; every reconstructed normalization stage matches CPU bytes. [Counterfactual result](key-counterfactual.json).

This establishes a local cause for the saved warm key-normalization error. It does not prove that a compiled correction will restore the cache or full-model outputs. Those remain native validation gates. Cold inputs already differ upstream, and cold attention-finish drift remains unresolved.

## Next milestone

[The next plan](next-experiment.json) proposes a separate bounded power-of-two wrapper: scale denominator by1/8 and seed by8, use the existing helper in its tested domain, then scale the corrected result by1/8. The proposed `[4,64]` denominator range covers this site's captured inputs. Exact scaling, neighbor mapping, power boundaries, fallback behavior and CPU preservation must first pass host gates, then a rank4 native supplied-seed component, then a controlled one-additional-site group candidate.

Full-model GPU numerical/lifecycle/partition/boundary/generation qualification and performance testing remain open. Actual device placement is unverified; CPU_AND_GPU denotes permitted compute units. Both qualified CPU model cohorts and pinned runtime sources remain unchanged: [final preservation](cpu-preservation-final.json). No further speed or accuracy improvement is claimed from these output-only diagnostics.

[Independent review](review.md), [outcome and hashes](outcome.json), [current gates](progress.json), [commands](commands.md), [checksum verification](checksum-verification.json), [handoff checksums](SHA256SUMS).
