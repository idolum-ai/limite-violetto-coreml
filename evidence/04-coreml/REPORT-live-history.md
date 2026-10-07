# Complete Core ML conversion — live report

Status: IN PROGRESS. Daniel authorized the next move after CPU cache qualification.

## Gates and approach

- G4.1: pending full unquantized model conversion/compilation/generation.
- G4.2: preserve the existing real-model limits (score max absolute 0.04; hidden relative RMSE 0.01; cosine >=0.995). Conversion changes will be measured separately from graph execution error. No acceptance widening.
- G4.3: pending request isolation, stops, cancellation, streaming and bounds checks.
- G4.4: pending reproducible invocation and manifest/artifacts.

First inspect converter dtype/state support and implement a graph-friendly complete forward path. Use bounded shards if needed to keep compilation and memory manageable. Every neural component must execute through Core ML for a complete-model claim; host code can handle tokenization, positions, orchestration and sampling. Preserve failures and original CPU reference. Original NVIDIA-runtime fidelity remains unverified.

## Dtype and rounding probes

Installed coremltools 9.0 exposes FP16/FP32 floating tensors and enforces FP16 state buffers. A finite-range BF16 rounding formula matches all 50,517 Torch reference values, but its first Core ML conversion fails. Intermediate inspection shows the runtime power expression returning exponent-squared rather than the expected power of two. This is an observed tool/runtime discrepancy, not yet an attributed upstream bug. The failed package and debug outputs are retained. Testing a precomputed power-of-two lookup to avoid that expression.

The lookup version passes all 50,517 rounding values exactly on both compute settings. A four-layer real-weight graph converts. Dynamic padding is unsupported; concatenating fixed zeros and slicing works. CPU execution succeeds, but GPU loading/execution aborts on a dynamic cache-read shape. Testing a fixed-capacity read with causal masking. The complete Torch graph candidate already fails the unchanged score gate on a 32-token prefix (max absolute 0.25424 vs 0.04), before Core ML conversion; final hidden relative RMSE 0.02327 vs 0.01. Retained in attempt1. These are failures, not qualified model results.

All 14 packages have converted: embedding, twelve four-layer stateful groups, and tied output head. Fixed-capacity cache reading resolves the observed GPU abort in the shard test. The graph has no weight quantization; source BF16 constants are stored exactly as FP32, with explicit BF16 rounding points and FP16 KV state. The projection ablation finds BF16-versus-FP32-then-round differences in all 48 layers on the tested source activations; FP16 cache narrowing changes small values in layers 36 and 39 without nonfinite values. Full-pipeline validation and bounded generation are next.

Full CPU pipeline validation: 54 comparisons completed. The numerical gate FAILS: maximum score error 0.3350806 exceeds 0.04, although all 15 tested next-token top-1 choices match. All 17 request lifecycle/behavior checks pass, including exact independent-request repeats, cancellation invalidation/disposal, explicit stops, callback streaming, ownership, padded IDs and overflow rejection. Actual 2048-position execution and real streamed completion are running separately.

The complete CPU Core ML math run finishes with boxed 391 after 447 generated tokens, stopping on explicit EOS151643. Elapsed generation time was 90.42 seconds with overlapping diagnostics; this is not a controlled benchmark. The first long-context probe exited 139 with no Python error; retained under boundary-attempt1 and rerunning in isolation with per-position progress. Context-boundary qualification remains pending.

The isolated chunk-by-chunk boundary run reaches all 2,048 positions with finite scores and rejects overflow without changing request state. Window-crossing score errors remain failing: 0.41047 at position 1024 and 0.64559 at position 1025. The initial exit-139 cause is not established; a separate single-call long-prefill reproduction is planned to distinguish that call pattern from the successful chunked run.

The complete CPU_AND_GPU pipeline also runs all three prompts and teacher-forced continuations: 54 comparisons, maximum score error 0.3610411 (FAIL), all 15 next-token top-1 choices matching, and all 17 lifecycle checks passing. This describes permitted compute units; actual operator placement is unprofiled.
