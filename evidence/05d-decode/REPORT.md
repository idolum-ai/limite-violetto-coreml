# Warm decode cost profiling

Status: COMPLETE. The measured bottleneck is the repeated backbone-group prediction calls: 97.97% of profiled forward time. No model, runtime, numerical threshold or default selection was changed.

## Validation gates

| Gate | Result | Evidence |
|---|---|---|
| D1: protocol and provenance | Passing 05c loader/runtime identities; independently pinned historical score fixture; instrumentation review resolved before launch | [Review](instrumentation-review.md), [frozen profiler](profile-1/source-snapshot.py) |
| D2: exactness and lifecycle | 84/84 forwards match saved scores exactly; 24 matched control/profile pairs; 336 component timings; 10/10 requests disposed; normal exit | [Raw results](profile-1/results.json), [log](profile-1.log), [exits](process-exits.json) |
| D3: timing accounting | First-shape trial, prefill and initial warmup excluded; all profiled calls have expected 14-name order; component plus residual times cover the complete forward | [Summary](summary.json), [8 summary/rejection tests](summary-tests-final.log) |
| D4: actionable next experiment | Independent static inventory and reviewed, bounded single-query product ablation; captured group0 replays twice with exact outputs and all 8 KV states | [Static inventory](static-inventory/REPORT.md), [capture](group0-warm-capture/results.json), [next experiment](next-experiment.json), [review](next-experiment-review.md) |

## Measured costs

The 12 four-layer backbone groups consume 97.9705% of profiled forward time. The head takes 2.0086%, embedding 0.0144%, and everything outside the component timers 0.0065%. That residual includes profiler wrapper/list overhead. Component timers include Python input preparation, bridging, native Core ML prediction and returned-array conversion; this is not an internal operator trace.

Individual group medians are 198–213ms per call, with no single group dominating. Head median is 51.2ms. The repeated group family is therefore the useful optimization target; reducing work outside those calls has little available benefit.

| Prefix length | Uninstrumented decode median (s/token) | Uninstrumented throughput (tokens/s) | Backbone share of profiled time |
|---|---:|---:|---:|
| 32 | 2.571 | 0.391 | 97.89% |
| 128 | 2.431 | 0.404 | 98.00% |
| 512 | 2.630 | 0.367 | 98.02% |

Each row has 8 uninstrumented and 8 instrumented decode observations across two measured requests. Every matched prefix/step has one of each condition. The median paired profiled/control ratio is 1.019, but individual ratios range 0.781–1.292 and the median difference is 46.7ms. Host/repeat/order variation prevents a precise causal estimate of timer overhead. The approximately 98% component attribution is consistent across all three prefix lengths.

The host was on AC power; reported load averages changed from 3.60 to 5.76. No thermal or performance warning was reported, but caches and other host activity were uncontrolled. No other inference experiment ran concurrently. These measurements establish a profiling baseline for future paired optimization tests; comparison with earlier runs is not controlled.

## Selected next experiment

The source value-product helper calculates 16 independent two-row products and then trims to the number of real queries. For single-token decode, only the first product supplies a returned row. The proposed isolated group0 variant retains that exact two-row input geometry and omits the 15 unused products, while retaining the 32-row score/query path and all other arithmetic.

This is a hypothesis to test, not an established internal hotspot or speedup. Core ML may already elide or fuse these branches; graph changes can also change surviving kernel selection or rounding. The [reviewed plan](next-experiment.json) requires graph/spec inspection, a strict single-query warm-only contract, fresh-state exact replay, and paired timing before progressing. The predeclared performance go threshold is a repeatable 10% median group-call reduction across blocks. Any integration into the full runtime requires separate qualification.

The saved fixture contains a real group0 call at absolute position 32, both outputs, and all eight state arrays before and after the call. The full pipeline prefill/decode scores are exact. Two independent restored-state replays reproduce both outputs and all eight post-call state arrays. Its guard records 4 stateless and 26 stateful predictions with 30 GIL releases; the process exits normally. It is a correctness fixture, not a benchmark. Array dtypes/shapes and hashes are recorded alongside the NPZ files.

## Preservation and handoff

All qualified source/runtime/guard/validator identities and the complete source/compiled artifact set remain unchanged. [Preservation check](baseline-preservation.json). The current qualified model remains the [05c CPU loader](../05c-compiled/REPORT.md). GPU/ANE, original NVIDIA-runtime fidelity and broader task accuracy remain outside this milestone.

[Reproduction commands](commands.md), [checksums](SHA256SUMS), [verification](checksum-verification.json). The proposed optimization has not been built or adopted.
