# CPU conditional-score integration

Status: **COMPLETE — promoted as the recommended explicit Core ML CPU selection on the pinned host.** All correctness, performance and preservation gates pass. [Decision and exact selection](decision.json). The original qualified CPU baseline remains available as fallback.

| Gate | Requirement | Evidence/status |
|---|---|---|
| I1 | One model/state per group; original arithmetic, weights and state operations preserved | [Final group0 build](conditional-group0-3/results.json), [all 12 groups reviewed](all-groups-static-review.md): PASS |
| I2 | Exact group0 cold lengths 1–32, persistent transitions and saved warm states | [50/50 exact](group0-parity-3/results.json), including all eight state arrays; 147 guarded calls |
| I3 | All 12 groups plus unchanged embedding/head, sealed source/cache | [Build](full-1/build.json), [independent preflight](full-1-structural-preflight.json): PASS |
| I4 | Original numerical, lifecycle, partition, boundary and completed-generation gates | [Aggregate PASS](full-1/milestone.json) |
| I5 | Isolated whole-pipeline benefit, exact scores, variability review and explicit fallback | [Full benchmark](pipeline-abba-1/results.json), [independent audit](pipeline-independent-audit.json), [P32 confirmation](prefix32-confirmation-1/results.json): PASS |

The full candidate passes 54 short comparisons and 17 lifecycle checks, 97 partition checkpoints across 13 cases, six boundary/capacity comparisons and four lifecycle/repeat checks. All 142 hidden-state comparisons are exact. The completed response matches all 674 qualified-reference tokens, produces the correct answer 391, stops on explicit EOS 151643, passes all seven score audits and disposes state. All native qualification processes exit normally. [Short](short/validation-CPU_ONLY.json), [partitions](../04d-audit/conditional1-partitions/results.json), [boundaries](boundaries/results.json), [generation](../04d-audit/conditional1-generation/results.json).

The graph keeps one native model/state per group. Four eager score selectors in each group become lazy conditional branches. Original arithmetic operations retain their serialized bytes; no original matrix product, state read/write or weight changes. Branch-local identities supply typed local returns. The unchanged compiled loader uses the same short-prefix replay and request semantics. [Design](build-design.md), [qualification design](qualification-design.md).

## Whole-pipeline measurements

Eight fresh processes execute A–B–B–A twice, with 32/128/512-token prefills and three forced decode updates. Round two reverses prefix order. Loading, verification, state creation/disposal and score comparisons are outside timers. All 96 recorded score vectors match the sealed historical pilot in the independent audit. All 32 requests dispose; every worker exits normally with 50 stateless calls, 408 stateful calls and 458 GIL releases. [Protocol](benchmark-protocol.md), [raw result](pipeline-abba-1/results.json), [decision review](decision-review.md).

| Prefix | Baseline median decode | Conditional median decode | Decode latency reduction | Prefill latency change |
|---|---|---|---|---|
| 32 | 2.354 s | 1.947 s | 17.28% | +0.87% |
| 128 | 2.342 s | 1.933 s | 17.46% | +0.74% |
| 512 | 2.354 s | 1.927 s | 18.10% | -5.76% |

Pooled median decode latency reduction is **17.83%**. Every prefix exceeds the predeclared 10% reduction gate; all prefill changes stay within the 10% regression limit. Each arm/prefix has four prefill and twelve decode observations. These are local observations, not a general latency guarantee.

The original benchmark records substantial P32 round variation: 8.06% reduction in round one versus 20.12% in round two. Telemetry also records an AC-to-battery transition between two optimized workers in round one; this does not prove the cause. Ordinary OS background work and caches remain uncontrolled. The original result and variation flag remain intact.

## One-shot short-context confirmation

A separately frozen P32-only A–B–B–A confirmation repeats the same excluded warmup and fresh measured request. It passes on its only attempt: median decode falls from **2.336 s to 1.818 s (22.18%)**; median prefill falls from **6.662 s to 6.044 s (9.27%)**. All 16 score vectors exactly match the historical capture, including shape, dtype and bytes. All eight requests dispose; each worker records 16 stateless calls, 96 stateful calls and 112 GIL releases, with normal exits.

All eight sampled timing boundaries report **Battery Power** and standard negative thermal/performance-warning status. This supports the bounded promotion decision while preserving the original variability caveat. It does not prove continuous host stability or generalize performance to every workload. [Frozen protocol](prefix32-confirmation-protocol.md), [result](prefix32-confirmation-1/results.json), [independent audit](prefix32-confirmation-independent-audit.json).

## Integration and retained evidence

The [recommended compiled command](../../README.md#run-the-experimental-core-ml-model) selects `artifacts/05g-integration/full-1/source` and its sealed compiled cache. Candidate 6 and its original compiled cache remain the explicit fallback. Both cohorts and all pinned runtime/validator/build-source identities pass [final preservation verification](final-preservation.json). No frozen runtime or acceptance criterion changed.

The first graph compiled but failed native loading because its false branch returned an outer value directly; local identity returns fixed that structural defect. The first batch then found that group6 has no identity operation to use as a template. The final builder uses an identity protobuf from sealed group0 in that case. All groups were rebuilt with one builder identity, and final group0 parity was repeated. Earlier attempts remain intact.

This is bounded CPU qualification on the pinned local environment. GPU/Neural Engine and clean-machine support remain unqualified. The separate full original-model GPU-policy screen [fails numerical criteria](../06b-full-gpu/REPORT.md). Portable CPU remains the practical inference recommendation; this promotion improves the qualified Core ML CPU research path.

[Reproduction commands](commands.md), [whole-project handoff list](SHA256SUMS), [handoff verification](checksum-verification.json).
