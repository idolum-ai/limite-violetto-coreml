# Step 3 — Real-checkpoint portable model

Historical initial results follow. The [subsequent cache investigation](../03b-cache/REPORT.md) fixes and qualifies the CPU path under the same criteria; the original failures below remain preserved. Metal remains unqualified.

Implemented on 2026-09-22. **Validation is partial: the full-prefix model passes local source comparisons; cached numerics fail the frozen limits.** This is a PyTorch implementation, not the complete Core ML port. Original NVIDIA-runtime fidelity remains unverified.

## Gate results

| Gate | Result | Evidence |
|---|---|---|
| G3.1 Exhaustive weight mapping | **PASS** | [Download identity](checkpoint-download.json), [743-tensor mapping](run3/weight-mapping.json) |
| G3.2 Independent numerical baseline | **BLOCKED** for original CUDA/vLLM outputs; **PASS** for the narrower local source comparison | [CPU results](run3/results.json), [Metal results](run2/results.json), [original head-method execution](head-and-stops.json) |
| G3.3 Cached/full-prefix agreement | **FAIL** on CPU and Metal at unchanged numerical limits | [Acceptance rules](ACCEPTANCE.json), [CPU calibration](calibration-cpu.json), [Metal calibration](calibration-mps.json), [CPU run](run3/results.json), [Metal run](run2/results.json) |
| G3.4 Input, stop and reset behavior | **PARTIAL**: input/reset/stop mechanics pass; published and observed EOS disagree. Explicit extra-stop option works. | [35 CPU checks](run3/results.json), [stop-loop tests](head-and-stops.json), [EOS observations](eos-observation.json), [explicit-stop validation](explicit-stop-validation.json) |

## What was implemented

The strict loader in [portable/model.py](../../portable/model.py) loads the pinned actor checkpoint into the published value-model backbone, preserving the actor's weights and dtypes. It rejects wrong checkpoint hashes, missing/unexpected keys, incorrect shapes/dtypes, and non-finite weights. No critic weights were used. Every load in the recorded model runs rechecks SHA-256.

- Checkpoint: `paradigma-inc/limite-1b-violetto@9402e422e87b220507963cd42c997459c1e87b43`.
- SHA-256: `9dc7e89e30acf473629f98ac3ade535fb05cc044fe7e399b78d0bc957283a5f5`.
- File size: **2,070,811,520 bytes**; download completed in approximately 186 seconds.
- **743 tensors / 1,035,254,176 parameters**: 338 BF16 tensors and 405 FP32 tensors. All elements finite.
- All 288 scalar name/shape mappings are explicit. FP32 XSA, gates and mixer storage is retained; arithmetic casts follow the source. Projection scales are folded while matrices remain BF16. RMS normalization uses the verified FP32 opmath epsilon.
- Output head uses the same embedding weight directly, BF16 projection and the published FP32 sigmoid score transformation. The pinned original runtime's `compute_logits` method was extracted without modifying it and executed on real CPU hidden states/weights: exact agreement with the portable head. This does not execute the original serving runtime.
- Request-owned BF16 K/V caches have a 2,048-token capacity. Overflow is rejected, not truncated. Reset discards storage; partially executed forwards invalidate the cache until reset. Only written cache positions are read.
- [portable/generate.py](../../portable/generate.py) adds bounded greedy or seeded sampling, excludes 13 padded vocabulary rows, and defaults to model EOS **151645**. An explicit `--stop-at-tokenizer-eos` option also accepts **151643** without changing the published config or default. JSON records the stop set, completion policy, and `eos`, `stop_token`, or `length` finish reason. Sampling uses the published temperature 0.6 and top-p 0.95. Streaming delivery and a complete Core ML runtime remain Step 4 work.

## Full-prefix and cache results

The reference is the **unmodified pinned upstream value-model backbone with the actual Violetto actor weights**. The adapter's uncached and cached attention paths are compared to its eager full-prefix path. They share weights and some upstream modules; this is a strong local implementation comparison, not an independent original-model oracle.

Each backend was tested on the three canonical prompts (44, 58 and 48 tokens), a fixed 1,026-token sequence, and a 12-token common teacher-forced continuation. Captures cover all 48 layer outputs, normalized embeddings, both attention/residual mixer outputs, final hidden states, and scores. The long case uses four 256-token prefills followed by individual tokens at positions 1024 and 1025. Numerical captures are in [CPU artifacts](../../artifacts/03-portable/run3/) and [Metal artifacts](../../artifacts/03-portable/run2/).

Acceptance rules were written before cached-model execution. Reference-only repeat and one-future-token shape probes observed zero difference on each backend. The predeclared calibration rule therefore froze the score max-absolute limit at **0.04** and hidden relative-RMSE limit at **0.01**, with minimum cosine **0.995**. Those limits were **not widened after failures**. This calibration samples only limited shape variation and does not establish a general BF16 error budget.

| Measurement | CPU | Metal / PyTorch MPS |
|---|---:|---:|
| Full-prefix comparisons | 220 / 220 pass | 220 / 220 pass |
| Full-prefix maximum absolute difference | 0 | 0 |
| Cached/teacher-forced comparisons | 185 / 247 pass | 178 / 247 pass |
| Worst cached score absolute difference | 0.4138155 | 0.2865295 |
| Worst cached hidden relative RMSE | 0.0212835 | 0.0252965 |
| Minimum cached hidden cosine | 0.9997782 | 0.9996808 |
| Leading-token agreement in recorded cache-score comparisons | 19 / 19 | 19 / 19 |
| Peak Python-process RSS | 3.54 GiB | 3.35 GiB |

The 19 score comparisons include the last window position twice (boundary check and final-score check). Numerical differences include all 151,680 head rows; generation excludes the 13 padding rows. RSS is process memory, not total system/GPU service memory. Timings in raw records are investigation observations, not isolated performance benchmarks or Core ML predictions.

The mask check confirms that local attention includes key 0 at query 1024 and excludes it at query 1025; global attention retains it. Reset, reuse for a different request, invalid IDs, overflow-before-mutation, wrong cache ownership, interrupted-forward rejection, and reset after interruption pass. These behavioral successes do not override the numerical gate failure.

## Diagnosing the failure

An initial CPU probe with a 43-token prefill plus one token happened to match exactly. The broader multi-step suite disproved any inference that CPU cached numerics were thereby qualified. Both the successful narrow probe and the failing broader runs are retained.

[First-layer Metal probes](mps-first-layer-diagnosis.json) found exact Q/K/V projections for the inspected positions but up to `7.63e-6` FP32 attention differences when the query dimension changed from full-prefix to one row. At one position this crossed a BF16 rounding boundary. This demonstrates one source of shape-dependent variation. **Inference:** rounding differences can accumulate through the 48-layer network; this evidence does not prove they are the sole cause or that the observed final error is acceptable.

The first validation attempt stopped in diagnostic code because it tried to create FP64 tensors on MPS. The harness now transfers to CPU before calculating FP64 comparison statistics. [Original failure log](run1.log) is preserved. The earlier MPS calibration used FP32 cosine reduction and contains small values above one from numerical reduction error; its difference metrics are zero, and its frozen limits are unaffected.

## Generation observations

The three 192-token Metal samples and the three 512-token CPU samples all reached their token caps. They contain the correct arithmetic values within reasoning, but **all six are incomplete responses**, not completed-answer successes. Raw text, token IDs, per-step timings and finish reasons remain in their run JSON files.

A separate clean 1,024-token greedy CLI run also reached its cap. Sampling with published temperature/top-p and seed 0 reached both the 1,024-token cap and the full 2,048-token context budget (44 prompt + 2,004 generated tokens). None emitted model-configured EOS151645. These runs are retained, including the repeated text.

Inspecting raw token IDs revealed an important distinction: the greedy arithmetic answer first emits **tokenizer EOS151643 after 559 generated tokens**, and the sampled answer does so after **500 tokens**. Both have already closed `</think>` and produced a correct final `\boxed{391}` at that point. Ignoring this token under the published EOS151645 policy allows further repeated text. The 512-token integer-roots sample also emitted tokenizer EOS, first at token 389. [Full token-level evidence](eos-observation.json) retains the first answer ending, including special tokens that clean text decoding hides.

The explicit extra-stop option was then tested in a fresh CLI request. It produced the same 500-token prefix as the earlier sampled run, answered **391** correctly, and ended with `finish_reason="stop_token"`, `complete=true`, and recorded stop IDs `[151643, 151645]`. [Result](cli-math-explicit-stop.json), [validation](explicit-stop-validation.json). This is a successful local response under a deliberately declared stop policy, not proof that the original runtime uses that policy. The default remains unchanged pending resolution of the conflicting published assets.

This generation-only option was added after the full suites; their original code snapshots remain under `artifacts/03-portable/run2/code/` and `run3/code/`. The new branch was validated through controlled stop-loop tests and the actual CLI request. Model arithmetic and frozen numerical limits were unchanged.

EOS, length termination, the distinction between model EOS and tokenizer EOS, and cleanup after a callback interruption also pass controlled generation-loop tests. These tests verify the loop independently of which tokens the model chooses.

## Reproduction and next decision

Run commands from the project root; use a new attempt name to avoid overwriting runs. [commands.md](commands.md) records commands and limitations. Upstream source files were left unchanged. [Summary](summary.json) is the machine-readable handoff, and [SHA256SUMS](SHA256SUMS) covers the final investigation files.

The next work is to resolve or properly qualify cached numerical variation before treating this as the reference for full-model conversion. Localize the first differing operation at later layers, test consistent reduction shapes/precision, and expand reference-only variation measurements. Any new numerical policy must be documented as a new experiment; it cannot retroactively pass these runs. Resolve the published-versus-emitted EOS discrepancy against original-runtime behavior. Original-runtime captures are still needed for fidelity to the published serving model. Step 4 has not begun.
