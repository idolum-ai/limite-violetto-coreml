> Follow-up: the cleanup crash described below is now diagnosed and mitigated by a pinned local guard. See the [current reliability report](../04b-reliability/REPORT.md). Numerical correction continues in the [precision report](../04c-precision/REPORT.md). The results below retain the original Step 4 measurements.

# Complete Core ML conversion — results

The complete real-weight model now converts, compiles, and generates through Core ML. A sampled CPU run answered `17 * 23` with **391**, completed after 447 tokens, and stopped using the explicitly enabled tokenizer EOS. **Numerical fidelity fails the unchanged acceptance gate. Reliability qualification remains partial because one native long-context crash is unexplained.** This is a working experimental conversion, not a qualified replacement for Violetto.

[Machine-readable summary and package hashes](summary.json) · [Reproduction commands](commands.md) · [Live experiment history](REPORT-live-history.md) · [Operator trail](../../OPERATOR-TRAIL.md)

## Validation gates

| Gate | Result | Evidence |
|---|---|---|
| G4.1 Complete model converts, compiles and generates | PASS on CPU; complete GPU-permitted pipeline also executes | [Completed generation](generation-cpu.json), [CPU](validation-CPU_ONLY.json), [CPU_AND_GPU](validation-CPU_AND_GPU.json) |
| G4.2 Numerical fidelity | FAIL; no threshold changes | [Precision diagnosis](precision-diagnosis.json), [boundary](boundary.json), [same-seed generation comparison](generation-comparison.json) |
| G4.3 Isolation, cancellation, stops, streaming and bounds | PARTIAL: all recorded lifecycle checks pass; initial native crash unresolved | [CPU checks](validation-CPU_ONLY.json), [GPU-permitted checks](validation-CPU_AND_GPU.json), [long-prefill isolation](longprefill-isolation.json), [first crash](boundary-attempt1/result.json) |
| G4.4 Reproducible local invocation and artifacts | PASS for this recorded environment; clean-machine installation untested | [Commands](commands.md), [manifest](../../manifest.json), [checksums](SHA256SUMS) |

Original NVIDIA/vLLM fidelity is still unverified: no independent original-runtime output capture was acquired. The qualified local CPU implementation is the reference used here.

## What runs

The pinned 1,035,254,176-parameter checkpoint is represented by **14 packages**: token/value embedding, twelve stateful groups of four transformer layers, and the tied output head. Every neural calculation runs in Core ML, including both residual mixers, all 48 attention/MLP layers, normalization and score transformation. The host performs tokenization, positional index construction, pipeline orchestration, sampling and text decoding.

The active packages total **4,952,393,711 bytes (4.61 GiB)**. They preserve source BF16 weight values as FP32 constants and retain FP32 scalar behavior where required; no weight quantization was applied. Explicit BF16 rounding is inserted at the source's rounding points. This is **BF16 emulation over FP32 arithmetic**, not native BF16 execution. Core ML requires FP16 state buffers, so the 96 KV buffers use FP16 (96 MiB of logical cache payload at capacity, excluding runtime copies and overhead). Each request owns twelve independent Core ML state objects.

Context is capped at 2,048 tokens. Prefill uses chunks of at most 32. Attention reads fixed-capacity caches with causal and inclusive local-window masking; no context is silently truncated. Input validation rejects padded token IDs. Generation excludes padded head rows, streams text to stderr, records JSON on stdout, and disposes request state on completion or cancellation. The default backend is `CPU_ONLY`.

`CPU_AND_GPU` denotes permitted compute units. Actual per-operator device placement was not profiled, and no Neural Engine claim is made.

## Numerical results

The short suite contains the three pinned prompts, selected group-boundary intermediates, and four teacher-forced continuation steps per prompt. It compares 54 tensors/scores per setting; it does not expose every individual Core ML layer output.

| Setting/test | Maximum score error | Existing limit | Next-token top-1 agreement |
|---|---:|---:|---:|
| CPU_ONLY short suite | 0.335081 | 0.04 | 15/15 |
| CPU_AND_GPU short suite | 0.361041 | 0.04 | 15/15 |
| CPU local-window position 1024 | 0.410471 | 0.04 | Not evaluated in boundary probe |
| CPU local-window position 1025 | 0.645587 | 0.04 | Not evaluated in boundary probe |

Only 30/54 CPU comparisons and 21/54 GPU-permitted comparisons satisfy their numerical criteria. Matching top-1 choices does not override score/hidden-state failures. Original criteria and their SHA256 are retained in the summary; nothing was recalibrated to admit the candidate.

The graph-friendly Torch implementation already fails before Core ML conversion: on the recorded 32-token prefix, score error is 0.254244 and final hidden relative RMSE is 0.02327 (limit 0.01). Therefore the error is not solely a Core ML execution issue. That earlier dynamic-read experiment is preserved in [attempt1](attempt1/torch-graph.json).

The [operation-level probe](precision-diagnosis.json) feeds identical real source activations into the two arithmetic paths. Normalizations in this probe match exactly. Every layer has at least one projection result differing between native BF16 linear execution and FP32 linear execution followed by BF16 rounding. For example, layer 0 differs in 14 query elements and two key elements. FP16 cache narrowing changes one small key element in each of layers 36 and 39; all checked cache values remain finite. These findings identify distinct numerical differences but do not prove a complete causal decomposition of final score error.

The standalone lookup-based rounding graph itself matches **50,517/50,517** tested values exactly on Torch, Core ML CPU, and Core ML CPU_AND_GPU. Its scope is finite model-scale values and rounding ties; it is not a general NaN/Inf/subnormal BF16 implementation. See [rounding results](rounding-probe-v2.json).

## Generation and behavior

The [Core ML run](generation-cpu.json) and [qualified CPU reference](generation-comparison.json) use identical prompt IDs, seed 0, temperature 0.6, top-p 0.95 and explicit stop set `[151643, 151645]`.

- Core ML: 447 generated tokens, completed, final boxed 391.
- Qualified CPU reference: 674 generated tokens, completed, final boxed 391.
- Their first 160 generated tokens match; they diverge at zero-based index 160. The complete sequences are not equivalent.

This single completed answer establishes a math smoke result, not broad mathematical reliability. The tokenizer/config EOS disagreement is unchanged: published actor EOS is 151645, while completed responses emit 151643. The explicit CLI option adds the latter and records the policy.

Both compute settings pass all **17 lifecycle checks**: exact fresh-request repeats for three prompts, disposal, invalid/padded/empty input rejection, wrong ownership, overflow, interruption invalidation, rejection after close, explicit stop handling, token callback delivery, and cancellation cleanup. Actual streamed CLI text and the final JSON were retained in [generation-cpu.log](generation-cpu.log) and [generation-cpu.json](generation-cpu.json).

An isolated run writes through all 2,048 positions and rejects the next token without mutating valid state. A second isolated run uses long single-call prefills and also reaches 2,048 with finite scores. These are execution checks; boundary numerical fidelity still fails.

## Failures and reliability boundary

All consequential failed experiments are retained:

1. Runtime `pow(2, exponent)` in the first rounding graph produces incorrect intermediate results; the debug output shows exponent-squared values. A precomputed power-of-two lookup passes the rounding test. This is an observed tool/runtime discrepancy, not an attributed vendor bug. [First result](rounding-probe.json), [debug](rounding-debug.json), [failed package](../../artifacts/04-coreml/rounding).
2. Two dynamic-padding forms are unsupported by the converter. Fixed zero concatenation followed by slicing converts. [Attempt 1 log](convert-0-attempt1.log), [attempt 2 log](convert-0-attempt2.log).
3. A four-layer shard with dynamic cache reads runs on CPU but aborts under CPU_AND_GPU with `NDArray dimension length > INT_MAX` (exit 134). Fixed-capacity reads avoid that observed abort. [Failure](attempt1/runtime-result.json), [fixed shard results](shard-runtime.json), [failed package](../../artifacts/04-coreml/model/0-dynamic-read.mlpackage).
4. The first full long-context probe exits 139 without a Python traceback while another Core ML process is generating. Its exact failure location and cause are unknown. Both subsequent isolated boundary/capacity runs succeed, including the original long-prefill call pattern. This does **not** prove concurrency caused the crash or that the issue is resolved. [Failure record](boundary-attempt1/result.json), [chunked isolation](boundary.json), [long-call isolation](longprefill-isolation.json).

G4.3 stays partial because of the last issue. Avoid presenting this as production-ready or as a tested concurrent-serving system.

## Reproduction and next decision

The [commands](commands.md), [manifest](../../manifest.json), [package inventory](summary.json), and [SHA256 list](SHA256SUMS) identify the scripts, inputs and outputs. Source integrity was rechecked: all 17 runtime files, nine model assets and eleven additional reference files pass. [Verification](source-verification.log).

The completed Core ML generation took 90.42 seconds; the CPU reference comparison took 58.35 seconds. These runs had different generated lengths and were not a controlled benchmark. Compilation, local work, memory pressure and scheduling were not controlled sufficiently for a performance recommendation. Step 5 remains unstarted.

The next useful work is a bounded precision follow-up: capture the first divergent projection under identical activations, separate projection/reduction arithmetic from cache narrowing and backend effects, and test a replacement only against the existing gates. Separately, seek a minimal reproducer for the unexplained long-context native crash. Do not optimize or package this candidate as fidelity-qualified, and do not relax acceptance to convert these failures into passes.
