# Equal-input normalization diagnosis

The warm normalization drift is localized to the reciprocal after the square root. CPU_ONLY and CPU_AND_GPU produce identical input, squares, ordered sums, means, epsilon additions and square-root denominators. The next operation, `1 / denominator`, differs at three values. Those differences propagate through multiplication and quantization, producing the previously observed 19 normalized-value differences.

This is a controlled localization on saved input/cache from one 44-token prompt, split into 32 and 12 tokens. CPU_AND_GPU is a compute policy; actual device placement remains unverified. No arithmetic replacement or full-model GPU qualification is claimed.

## Validation gates

| Gate | Outcome | Evidence |
|---|---|---|
| N1: output-only clone preserves existing model | PASS: all 46,035 candidate operations, weights, endpoints and state interfaces unchanged; 12 outputs added | [Design](taps-design.md), [preflight](preflight.json), [compiled build](taps-1/results.json) |
| N2: native diagnostic controls | PASS: 8 calls, 4 policy/chunk cells; original endpoints and all eight states match frozen prior results and tapped endpoints/states match originals | [Protocol](replay-1-protocol.json), [replay](replay-1/results.json) |
| N3: equal-input operation localization | PASS: warm inputs exact through sqrt; reciprocal first differs | [Independent review](review.md), [numerical analysis](normalization-analysis.json) |
| N4: arithmetic correction | NO-GO in host preflight; native run held | [Rejected candidate and failures](reciprocal-host-1/host-preflight.json) |
| N5: cold attention-finish interval | Still open; cold normalization input already differs | [Prior six-tap result](../06d-gpu-rounding/residual-replay-3/results.json) |
| N6: CPU preservation and handoff | Both CPU cohorts/source preservation PASS; independent audits complete | [Preservation](cpu-preservation.json), [seal](checksum-verification.json) |

## Controlled findings

All eight predictions complete normally with eight state calls and eight GIL releases. Four independently restored policy/chunk cells pass exact original-versus-saved and tapped-versus-original endpoint/state comparisons. Overlapping `value_17` and `x_69` arrays match the earlier six-tap experiment byte-for-byte under both policies and chunks. Exposing these intermediate outputs therefore preserves the observed endpoints and cache behavior.

The warm chunk gives the causal boundary:

| Stage | CPU versus GPU policy |
|---|---|
| Attention residual `value_17` | Byte-exact |
| Square `x_65` | Byte-exact |
| Ordered sum `var_5669` | Byte-exact |
| Mean `mean_7`, epsilon addition `var_5672` | Byte-exact |
| Square root `denom_7` | Byte-exact |
| Reciprocal `var_5675` | 3 values differ; maximum absolute difference 2.9802322387695312e-8 |
| Raw normalized value `x_67` | Differs |
| Quantization step `step_47` | Byte-exact |
| Quotient `var_5690` | Differs |
| Explicit-even round `var_5691`, normalized output `x_69` | 19 values differ, at query row 8 / absolute position 40 |

The first reciprocal mismatch is at warm `[0,1,0]`: CPU 0.37130776047706604 versus GPU-policy 0.3713077902793884. The first normalized output mismatch is later at `[0,8,23]`: CPU -0.87890625 versus GPU-policy -0.8828125. A small reciprocal discrepancy need not alter every rounded row; the later quantization boundary determines which values change.

Cold normalization cannot independently establish this equal-input chain: its attention residual already differs at 146 values in query row 19. The first cold mismatch remains earlier in the attention-finish interval. Do not attribute all remaining model drift to the reciprocal or assume this one correction will solve both trajectories.

## Reproduction and scope

[Commands](commands.md), [independent review](review.md), [gates](protocol.json), [operator trail](../../OPERATOR-TRAIL.md). No qualified CPU runtime, selection, source package, compiled cache, weight, or acceptance threshold changed. The successful 06d rounding correction remains a separate local candidate; it is not promoted into the CPU runtime or a complete GPU model.


## Reciprocal and rounding analysis

The independent audit finds the warm reciprocal discrepancy at rows 1, 2 and 8: each GPU-policy result is one FP32 ULP above CPU. The CPU reciprocal matches the nearest-FP32 reference at all 12 warm denominators. Exact rational distances to adjacent FP32 values independently verify the reciprocal references; this makes no claim about the correctness of the preceding square root itself.

The resulting quotient differs at 3,375 warm values across those three rows, but only 19 rounded outputs differ, all in row 8. In those 19 cases CPU is just short of a half-integer and GPU policy lands exactly on it; there are zero strict half-boundary crossings. For example, `[0,8,23]` has quotient CPU -225.49998474121094 versus GPU-policy -225.5. Correct ties-to-even gives -225 versus -226, explaining the output difference even after the earlier rounding-rule correction. Thus this is now an upstream arithmetic discrepancy, not evidence that the explicit-even rule has regressed.

Cold normalization has unequal input at row 19, but among its 31 equal-denominator rows, 11 reciprocal results also differ. That shows the reciprocal discrepancy is not confined to the warm chunk; it does not explain the earlier cold attention-residual divergence. [Raw-array analysis](normalization-analysis.json), [independent control recomputation](residual-summary-1.json).

A separate bounded arithmetic probe was prepared. It compares the original reciprocal with one compensated Newton correction on captured denominators and a directed positive-finite set. It first requires the isolated control to reproduce saved results and the diagnostic to preserve that control. Nonreproduction prevents a correction claim. Native candidate accuracy, original CPU byte preservation and model reinsertion are separate gates.

The [offline counterfactual](normalization-counterfactual.json) independently reproduces the saved native multiply, quotient, round and final-multiply stages byte-for-byte under each policy. Substituting only the saved CPU reciprocal into the warm GPU-policy arithmetic then matches all four CPU stages exactly and removes all 19 normalized-output differences. This establishes a sufficient local explanation in the captured arithmetic, not an implemented native correction or full-model result.


## Rejected correction candidate

The proposed compensated Newton refinement fails its host prerequisite on three directed inputs: the FP32 values immediately below 1, 2 and 4. The original CPU reciprocal is correctly rounded, but refinement changes it to exactly 1, 0.5 and 0.25 respectively instead of the next representable value above. All captured denominator cases pass, which shows why captured inputs alone are insufficient for this gate.

The native 16-call run is held: zero native predictions, no model conversion or compilation for this correction. The candidate code, design, inputs and failures are preserved. This is a correction-candidate failure, not a failure of the eight-call normalization localization or offline counterfactual. A revised candidate needs a fresh protocol and must pass both exact-reference and original CPU byte-preservation checks before native testing. [Host evidence](reciprocal-host-1/host-preflight.json), [candidate design](reciprocal-design.md).


Exact rational analysis further localizes the rejected implementation: the ideal Newton expression and a sequence of two correctly rounded fused multiply-add operations both match the reference on all 108 exercised values. The helper's first residual also matches, but its final compensated multiply-add loses the distinction at the three problematic boundaries. This identifies a limitation of this reused compensated implementation on these inputs, not a failure of the ideal formula or a native compiler cause. It does not prove the formula universally correct.

The next milestone is a separate reciprocal implementation that retains the necessary boundary information and passes the original failing cases, followed by native reproduction and group controls. [Next experiment and evidence slots](next-experiment.json). The cold attention-finish target remains open. [Outcome record](outcome.json), [handoff verification](checksum-verification.json).
