# Practical CPU evaluation — completed pilot

**Recommendation: use the portable PyTorch CPU path for practical inference today.** Keep sealed Core ML candidate 6 as the validated correctness baseline. In this fixed-stream pilot, Core ML prefill was 11.2–13.5× slower and median decode latency was 28.5–34.3× slower, while using about twice the process peak memory. The port is numerically qualified within its tested scope, but its current CPU implementation offers no measured performance advantage.

## Measured cost

| Metric | Core ML candidate 6 | Portable CPU |
|---|---:|---:|
| Fresh-process entry to ready | 238.24 s | 3.92 s |
| Peak process RSS, combined workload | 6.82 GiB | 3.51 GiB |

Warm prefill values below are the median of repeats 1–2 at each shape. Decode throughput is sixteen fixed single-token forwards pooled across those two trials. The first trial at each shape remains separate in the evidence.

| Input tokens | Core ML prefill | PyTorch prefill | Prefill latency ratio | Core ML decode tok/s | PyTorch decode tok/s |
|---|---:|---:|---:|---:|---:|
| 32 | 6.69 s | 0.60 s | 11.2× | 0.43 | 14.79 |
| 128 | 27.19 s | 2.19 s | 12.4× | 0.41 | 14.58 |
| 512 | 114.41 s | 8.45 s | 13.5× | 0.43 | 12.32 |

All **27/27 matched score comparisons pass**, with matching top1 selections and maximum absolute score difference 3.814697265625e-6 against the unchanged 0.04 limit. The speed comparison did not silently switch precision policy or substitute a different token stream. Request disposal checks pass.

Evidence: [comparison and raw ratios](comparison.json), [Core ML process](coreml-pilot-1/results.json), [PyTorch process](torch-pilot-1/results.json), [independent review](independent-review.md), [reproduction commands](commands.md).

## Small task check

Three new tasks were run on the qualified **portable CPU reference**, with the original tokenizer/template, seed0, temperature0.6, top-p0.95 and a 1,024-token output budget. Every task completed and produced the correct numeric final boxed answer after the reasoning terminator.

| Task | Final answer | Generated tokens | Result |
|---|---:|---:|---|
| arithmetic | 476 | 315 | Complete and correct |
| simple_algebra | 5/2 | 450 | Complete and correct |
| short_word_problem | 17 | 491 | Complete and correct |

[Full responses and input/output IDs](cpu-reference-1/results.json), [frozen task design](task-design.md), [parser checks](parser-self-check.json), [source snapshots](cpu-reference-1/snapshot-verification.json). These three easy tasks at one seed do not establish broad mathematical competence. They were not rerun through Core ML, so no token-identity claim is made for these prompts. Core ML's separate completed response still matches all 674 tokens of the qualified arithmetic reference and passes seven score audits: [existing generation qualification](../04d-audit/candidate6-generation/summary.json).

## Measurement boundaries

- One fresh process per backend, run sequentially; one first-use 32-token request, then three requests per shape with eight forced decode steps. Only the later two per-shape trials enter warm comparisons. This is a pilot, not a startup distribution.
- The exact same frozen 719-token stream supplies all prefixes and forced continuations. Sampling, tokenization, logging and score comparisons are outside the measured inference spans.
- Prefill includes input preparation, backbone execution and final score projection. Request/state creation is separately recorded, with combined first-score latency available. These are not full text time-to-first-token measurements.
- Startup includes imports, package/checkpoint identity verification and backend initialization. The `model_load_seconds` field also includes backend import and verification. Filesystem/compiler caches were not evicted; this is not proven cold-cache startup. No separate warm-reopen measurement was made.
- Peak RSS is the process lifetime high-water mark over startup and the combined workload. It is not per-prefix memory, process physical footprint, or total system/accelerator memory.
- Both runs used the same macOS26.5.2 arm64 host with32GiB RAM, Torch2.7.0 and four Torch threads; Core ML used CPU_ONLY. AC power and no thermal warnings were recorded. Initial one-minute host load averages were3.70 and4.34 on10 logical CPUs. Other desktop activity was not controlled; backends were not randomized across multiple processes. Raw ranges are retained, and no confidence interval or universal performance threshold is asserted.
- The results apply to this exact, arithmetic-preserving Core ML implementation. They do not prove that Core ML in general, GPU execution or the Neural Engine is slow. No accelerated-device inference occurred in this pilot.

## Next experiment

Investigate startup first with **one stateful group**: persist a compiled `.mlmodelc`, separately measure compilation, then fresh-process reopening and state creation with the installed `CompiledMLModel` API. Compare against the same sealed `.mlpackage` loading path. Bind artifacts to package and runtime hashes. No speedup is assumed from API availability.

The [read-only loader inspection](loader-inspection.json) confirms stateful compiled loading is supported but is not a drop-in runtime replacement: `CompiledMLModel` lacks the pipeline's `get_spec()` call. Preserve source specs or the required metadata separately. Prediction and lifecycle checks are required before any successful loading experiment expands into a new runtime candidate. Warm decode also needs separate component-level profiling; fixing startup alone cannot eliminate the measured decode gap. GPU/Neural Engine work remains a later qualified optimization.

## Validation gates

| Gate | Result | Evidence |
|---|---|---|
| G5.1 Repeatable cost measurements | PASS for the stated pilot scope | [Comparison](comparison.json), [commands](commands.md) |
| G5.2 Bounded task usefulness | PASS for three portable-reference tasks plus existing Core ML arithmetic fixture | [Tasks](cpu-reference-1/results.json) |
| G5.3 Local reproducible handoff | PASS for recorded local artifacts; clean-machine installation untested | [Summary](summary.json), [checksums](SHA256SUMS) |
| G5.4 Evidence-based recommendation | Use portable CPU today; optimize Core ML separately | Measurements and next experiment above |

Candidate6, its runtime, the acceptance criteria and the default package selection remain unchanged. The earlier Step4 checksum list is historical; this step's final checksum list covers the updated handoff.
