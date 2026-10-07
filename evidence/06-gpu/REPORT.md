# Original group0 GPU-policy probe

Status: COMPLETE — operationally successful, strict compute-policy exactness NO-GO; broader GPU qualification remains unmeasured. All five cases produce inexact `y` and seven of eight state arrays under CPU_AND_GPU. Their group-output relative RMSE remains below the existing 0.01 diagnostic limit, with cosine above 0.999968. This does not fail or pass the full-model score gate, which has not been run for this candidate under GPU policy.

| Gate | Result | Evidence |
|---|---|---|
| G1: same artifact and guards | PASS: original sealed group0 compiled graph, verified source/cache/runtime before and after; 2 stateless +18 stateful calls, 20 GIL releases | [Raw results](gpu-group0-1/results.json), [protocol](PROTOCOL.md), [prelaunch review](prelaunch-review.md) |
| G2: bounded cold/warm cases | PASS for execution: cold 3/32 tokens and warm positions 32/128/512; eight CPU controls valid; ten GPU-policy calls complete; all fresh state restorations exact | [Captures and comparisons](gpu-group0-1/results.json) |
| G3: exactness and diagnostic size | Exactness FAIL in all ten GPU-policy comparisons. All five `y` outputs satisfy the recorded relative-RMSE/cosine screen; no timing claim | [Independent array audit](summary.json) |
| G4: preservation and status | PASS: source, compiled artifact, runtime and pinned script/protocol/fixtures preserved; native process exits 0 | [Preservation](baseline-preservation.json), [exit record](process-exits.json) |

## Observed differences

| Case | `y` relative RMSE | `y` cosine | `y` max absolute error | Exact array gate |
|---|---:|---:|---:|---|
| cold-prefill-3 | 0.001353 | 0.999999110 | 8192 | FAIL |
| cold-prefill-32 | 0.001396 | 0.999999052 | 8192 | FAIL |
| warm-32 | 0.008293 | 0.999968581 | 128 | FAIL |
| warm-128 | 0.007185 | 0.999977033 | 96 | FAIL |
| warm-512 | 0.007891 | 0.999971376 | 96 | FAIL |

The absolute errors refer to unnormalized group outputs and are reported alongside relative errors. The auxiliary `tap23_out` output and first-layer value cache match exactly; the first-layer key cache and all six later-layer cache arrays differ in every case. This places a divergence by the first key-cache write, but does not identify its upstream cause. Earlier key-rounding findings are hypotheses to revisit, not a proven diagnosis of these new captures.

Warm calls start from qualified CPU-produced caches. This isolates a single GPU-policy call; it does not test accumulated GPU-produced caches over a full conversation. Whole-cache relative errors can also be diluted by unchanged/zero rows, so state-array error summaries must not be interpreted as active-row error bounds. Independent replay-array checks confirm both GPU-policy repeats agree exactly with each other on both outputs and all eight states in all five cases. The audit rehashed 54 evidence files and adds active-written-slot metrics to distinguish them from whole-cache averages. All arrays and first-difference locations are retained for diagnosis. State reads expose float32 arrays from source-declared FP16 states; equality means exact numeric values with matching shapes/dtypes, not raw-storage-byte equality.

## Placement and decision

Core ML's compute plan anticipates GPU for 19,429 operations and reports unknown usage for 23,069 others. This is a plan, not a runtime execution trace. CPU_AND_GPU permits GPU use; actual execution-device placement remains unverified. No latency benchmark or speedup claim follows from this probe.

The strict same-graph parity screen is stricter than the project's original approximate numerical acceptance criteria. Its failure is therefore not sufficient grounds to reject GPU viability. The next decisive GPU milestone is a bounded full-model short numerical screen under the unchanged score/hidden limits, including multiple decode steps with GPU-produced cache state. Only then decide whether to prioritize first-key-cache localization or broader GPU qualification. [Next GPU gate](next-experiment.json).

Separately, the [05f CPU warm-score ablation](../05f-scores/REPORT.md) achieved a 17.76–22.59% local group-call reduction with all 180 calls exact. It remains a candidate for future routing/state-ownership integration and full requalification; it is not installed in the runtime.

[Commands](commands.md), [checksums](SHA256SUMS), [checksum verification](checksum-verification.json). The qualified CPU runtime is unchanged. Neural Engine conversion, independent NVIDIA fidelity, clean-machine deployment and broader task accuracy remain outside these milestones.
