# Step 2 — Stateful Core ML component experiment

Date: 2026-09-22. **Completed. Recommendation: continue to a strictly loaded portable Violetto model, using the qualified upstream backbone.**

Six synthetic graphs—small and production-dimension local attention, global attention, and four-path MUDD mixing—converted, compiled, and executed through Core ML. All final component numerical and state checks passed the thresholds declared before those runs. An incorrect normalization-epsilon assumption was discovered and corrected. Fidelity to the real Violetto checkpoint remains unverified.

## Validation gates

| Gate | Result | Meaning |
|---|---|---|
| G2.1 Convert/compile/execute | PASS | All six final graphs executed with CPU/GPU allowed; Neural Engine disabled |
| G2.2 State correctness | PASS | Chunked prefill and repeated decode, full-prefix comparison, K/V contents, fresh-state zeros, B versus clean reference, A/B isolation, future/unused-slot masking, and window-boundary probes passed |
| G2.3 Numerical comparison | PASS for tested component arithmetic; model/BF16 fidelity unresolved | Final Core ML/FP32-reference errors meet unchanged limits. BF16 differences are diagnostic and are not covered by a model-fidelity threshold |
| G2.4 Resource feasibility | PASS for bounded component screen | Production attention warm calls under 5 ms; experiment process peak RSS under 0.66 GiB. These do not predict complete-model performance or total system memory |

The remaining G1.3 independent Violetto-output baseline is still unavailable. No checkpoint weights were downloaded, no mathematical answers were generated, and Steps 3–5 were not executed.

## What was tested

Code: [components.py](../../experiments/components.py), [runner](../../experiments/run_components.py), [sequential suite](../../experiments/run_suite.py), [upstream qualification](../../experiments/qualify_upstream.py). Upstream attribution and pinned sources are retained in `sources/` and the manifest.

Production attention uses hidden width 1,280, 10 query heads, 2 KV heads, head dimension 128, and a 2,048-slot cache. Each local/global run processes **1,031 tokens in 14 calls**, including chunks of 1, 7, 32, 216, and 256 tokens. The small case uses width 32 and a 16-slot cache. Production cache storage is 2 MiB for K+V **per component**, independent of activation/temporary memory.

The graph includes Q/K/V and output projections, value-embedding gates, post-VE V caching, Q/K RMS normalization, local rotary tables, global rotary bypass, XSA, and output gates. Inputs supply synthetic hidden activations and synthetic value embeddings; token embedding lookup, full decoder residual/MLP integration, the LM head, and generation are outside this experiment.

The mixer graph exercises both attention-input and residual-base branches at both real sites, layers 24 and 47, with three taps and production inner width 32. Sequence lengths 1, 7, and 17 were checked.

Synthetic weights and inputs use recorded seeds. Weights are rounded/folded to mirror BF16 constants, then represented in FP32. Arithmetic and I/O are FP32; K/V state is FP16. Original per-operation BF16 activation rounding is **not emulated**. Precomputed BF16-rounded rotary tables are used for the bounded position range. The persistent state path is explicitly visible in the saved MIL graphs.

## Critical numerical correction

The source comments and the initial Step 1 report suggested default RMS epsilon followed the input dtype. Actual local `torch==2.7.0` CPU tests contradicted that: FP16 and BF16 default normalization matched **FP32 epsilon, `1.1920928955078125e-7`**, exactly on the probe; explicitly supplying BF16 epsilon `0.0078125` produced very different results.

Evidence: [executable probe](../../experiments/rms_epsilon_probe.py), [probe output](rms-epsilon-probe.json). The pinned serving-version source also supports FP32 opmath/accumulation epsilon: [PyTorch 2.11 composite source](torch-2.11-layer_norm.cpp) and [CUDA source](torch-2.11-cuda-layer_norm_kernel.cu), particularly `_fused_rms_norm_cuda`. [Current official documentation](https://docs.pytorch.org/docs/main/generated/torch.nn.modules.normalization.RMSNorm.html) now states the opmath rule. No CUDA execution was performed here; the claim about 2.11 CUDA is source inspection.

The first suite retained the incorrect explicit BF16 epsilon and compared the converted graph against the same assumption. It passed internal conversion checks while differing from the native upstream BF16 probe by as much as **0.0752** in production attention. This demonstrates why conversion agreement alone is insufficient.

We corrected the implementation epsilon and reran all six graphs. [Acceptance v1](ACCEPTANCE-v1.json) and all old results are retained. [Acceptance v2](ACCEPTANCE.json) records the reason; **no numerical tolerance was relaxed**. The corrected reference calls the original upstream normalization without substituting our implementation. Remaining production FP32-versus-upstream-BF16 diagnostic differences are about **0.00228** for attention and up to **0.00429** for the mixer on these synthetic cases. These are not model-level acceptance limits.

Other probes in [upstream-qualification.json](upstream-qualification.json) found:

- Both upstream mixers produced bit-identical BF16 outputs on all four tested paths; rotary Q/K outputs were also identical.
- A BF16 projection weight multiplied by the zero-dimensional FP32 scale followed BF16 promotion and matched the serving fold. Keeping that weight FP32 before folding changed 49 of 1,024 elements, with maximum difference 0.015625.
- Narrowing XSA alpha to BF16 before FP32 tanh changed all ten test values. Preserve the real checkpoint's mixed parameter precision rather than copying the critic loader's blanket BF16 cast.

## Final measurements

| Final production graph | Convert | Compile/load call | Warm one-token attention call | Experiment process peak RSS | Maximum Core ML/reference output error |
|---|---:|---:|---:|---:|---:|
| Local attention | 0.442 s | 0.161 s | 3.80 ms | 0.655 GiB | 3.98e-6 |
| Global attention | 0.380 s | 0.139 s | 4.39 ms | 0.648 GiB | 2.47e-6 |
| Four mixer paths | 0.227 s | 0.077 s | Not measured as a repeated decode benchmark | 0.330 GiB | 4.17e-7 |

Final attention first-prediction calls were 117 ms local and 34 ms global. First use of the single-token shape took 30 ms and 9 ms; the warm medians exclude those first single-token calls. Some earlier runs had much higher first-use latency (up to 1.51 s), showing compilation/cache effects. These are observations from a working Mac after prior experiments, **not controlled cold-cache benchmarks**. No complete-model tokens-per-second estimate is inferred.

Peak RSS is Python-process high-water memory, including that process's reference and conversion work; it excludes separate Core ML service/compiler processes and does not measure total machine memory. CPU/GPU are permitted compute units; actual device placement was not profiled.

Maximum FP16 cache-value difference was 0.0009765625 for K and 0.000244140625 for V, within the declared limits. Chunked versus full-prefix PyTorch output error stayed below 3.2e-7. All outputs checked were finite.

## State and boundary evidence

The production local graph sees key zero at query position 1,024 and excludes it at 1,025; the global graph retains it at both positions. An injected key-zero value impulse yielded a nonzero output exactly where expected, and zero at the excluded local position. Both outputs agreed with the portable reference. This independently tests the inclusive boundary rather than only comparing two implementations of the same mask expression.

Additional checks verified all fresh-state entries are zero, new request B matches a clean full-prefix reference, B cannot mutate request A's state, poisoned future/unused cache slots have no effect, and changing a future token in a prefill chunk cannot affect earlier outputs. The protocol requires contiguous append-only input positions; this experiment is not a public API that validates arbitrary malformed request sequences.

## Conversion failures and resolution

All attempts remain in this folder. The final successful attention update uses contiguous dynamic slice assignment.

1. Direct scatter: converter saw FP32 state data and FP16 updates, rejecting the dtype mismatch.
2. Explicit functional scatter plus `copy_`: assignment pass rejected a copy with no matching slice.
3. Scatter plus slice assignment: symbolic expanded-index shape inference failed.
4. Contiguous dynamic slice assignment: conversion and prediction succeeded, but the test harness's FP16 NumPy input to `write_state` was rejected.
5. Using FP32 NumPy arrays for the Python state writer resolved the harness error; persistent state remains FP16, as shown by the compiled graph.
6. Epsilon correction required a new versioned suite; it passed the same numerical limits.

Successful final artifacts and complete machine-readable results:

- [Small local attention](attention-small-local-attempt6.json)
- [Small global attention](attention-small-global-attempt2.json)
- [Small mixer](mixer-small-attempt2.json)
- [Production local attention](attention-production-local-attempt2.json)
- [Production global attention](attention-production-global-attempt2.json)
- [Production mixer](mixer-production-attempt2.json)
- [Sequential commands and exit codes](suite-v2-execution.json)
- [Aggregate summary](summary.json)

Each result points to its `artifacts/02-components/…` directory with the Core ML package, traced graph, MIL text, and NPZ captures. Script hashes and acceptance hashes tie results to their inputs. NPZ files contain synthetic numerical outputs and can be read with `allow_pickle=False`.

## Next decision

Proceed to Step 3: acquire the pinned Violetto checkpoint, strictly map every real tensor into the qualified portable backbone, retain actual parameter dtypes, add its tied output head and score transformation, and compare full-prefix/cached results. Use the corrected explicit RMS epsilon. Continue pursuing attributable original-runtime outputs before claiming fidelity, and measure accumulated BF16/FP32 differences on actual activations before deciding the final Core ML precision policy.
