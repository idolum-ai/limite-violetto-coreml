# Cache numerical investigation — live report

Status: IN PROGRESS. Daniel authorized continuation and continuous logging.

The original Step 3 limits and failures remain unchanged. This follow-up will first isolate shape-dependent reductions using the real checkpoint, then test bounded implementation changes. No threshold changes or original-runtime fidelity claims are authorized by a local pass. EOS defaults remain unchanged; the explicit tokenizer-EOS option remains documented.

## Experiments

1. Test whether padding only attention query rows restores full-prefix reduction behavior while keeping cached projections/MLP execution. Compare original and padded modes against the unchanged upstream full-prefix calculation at the three canonical prompts and local-window boundary. Preserve all outputs and performance costs.
2. If query shape is insufficient, localize the first unequal operation and evaluate precision/reduction alternatives in separately named experiments. Do not relabel a changed-precision calculation as source-exact.

## Results

### CPU attention-shape ablation — complete

[cpu1 results](cpu1/results.json): the original cache reproduces prior score errors (0.337642, 0.223475, 0.115558 on the three prompts). Padding cached attention query rows to the current prefix length produces **exact full-prefix agreement** on all three prompts, across the final hidden sequence and scores. No weight/activation dtype or tolerance changed. This is preliminary; the long/window and full intermediate suites remain pending.

FP64 attention also eliminates within-mode cache drift on these prompts, but changes full-prefix outputs relative to the unmodified source (for example, score difference 0.152456 on prompt 0). Therefore it is a diagnostic ablation, not a source-exact replacement.

### EOS source evidence

The already-pinned `sources/value-model/README.md`, line 187, identifies endoftext/151643 as the completed rollout token and im_end/151645 as a prompt-turn delimiter. This supports the observed local EOS behavior but does not execute Violetto's original serving runtime. The earlier audit missed this relevant prose. Preserve the actor config conflict rather than erase it.

### Running

Test padded attention on CPU and Metal through the 1,026-token boundary sequence. Full Core ML conversion has not begun.

### Metal query-padding result — complete

[mps-window results](mps-window/results.json): padding query rows does not qualify Metal. Cached final-score errors remain 0.2660, 0.3651, 0.1168 on the short prompts and 0.2605 on the boundary sequence. The native errors are retained alongside them. CPU long/window validation is still running. No acceptance threshold was changed.

### CPU query-padding boundary result — complete

[cpu-window results](cpu-window/results.json): the padded-query CPU calculation matches the unmodified upstream full-prefix hidden sequence and final scores **exactly** on the 1,026-token input with four 256-token chunks and two single-token calls. This closes the preliminary long-input check. Full-prefix-size padding has a quadratic attention cost; testing smaller fixed query-row padding before promoting a default. The full 48-layer/teacher-forced gate suite still needs to run on the selected implementation.

### Fixed-row CPU ablation — complete

[cpu-small-rows results](cpu-small-rows/results.json): 8, 16 and 32 query rows give exact agreement on all three short prompts; four rows fail two prompts (score errors 0.1670 and 0.2703). Select **8 rows** for the next complete gate suite, clamped to the available key count so an uncached short prefix retains its original shape. Only CPU uses this candidate; Metal stays on the original path and remains unqualified. This changes reduction shape, not weights, dtypes, masks, or numerical criteria.

### Additional tiny-prefix stress — failure retained

[cpu-partitions.json](cpu-partitions.json) tests one-token initial prefill followed by tiny/irregular chunks on deterministic random valid IDs. Prefixes 1–4 pass; longer cases beginning from those tiny prefixes fail, despite the original gate cases passing so far. This is an additional edge case, not covered by the first three prompts. Investigating source prefix-shape invariance and whether very early cache state needs refresh. Do not yet treat the candidate as generally qualified.

### Source prefix-shape probe — cause localized

[prefix-shape-probe.json](prefix-shape-probe.json): the **unmodified source itself** produces different hidden states for the same early causal positions when a tiny prefix is extended to 127 tokens. Lengths 3/4/7/8/9 differ; tested lengths 16/32 match. Consequently, retaining early KV values across this shape transition can preserve the wrong rounded states. The original all-history stress also compared earlier returned rows to a later, numerically changed source prefix.

Candidate remedy: while the CPU cache contains fewer than 16 tokens, retain those token IDs and rebuild the small prefix when more input arrives; discard the temporary IDs at 16 tokens. Normal templated prompts already exceed this threshold, so their prefill/decode path is unchanged. Validate each new chunk against the source with **the same available prefix**, and retain the future-prefix difference as a diagnostic. Original Step 3 criteria/cases remain unchanged.

### Warmup plus eight rows — refined diagnosis

The original full suite passes all 467 comparisons with eight rows. Matched-prefix tiny tests now pass through length 16 and for the irregular 127-token input, but fail at the first incremental 16→17 transition. [tiny17-diagnosis.json](tiny17-diagnosis.json) shows that the source's first 16 states are unchanged here; it is the current query reduction: 8 rows fail, while **16 and 32 rows match exactly**. Select 16 query rows plus the bounded <16-token cache warmup for final validation. Retain all eight-row results.

### Sixteen-row extended stress — PASS

[cpu-partitions-v3.json](cpu-partitions-v3.json): all 12 tiny/irregular-prefix cases pass, with 137 matched-prefix comparisons. All compared hidden states and scores are **exactly equal** to the unmodified source at the same available prefix. The independent source's future-prefix variation remains in the diagnostic fields; it was not relabeled as cache error or erased. Running final full gate suite, warmup lifecycle tests, and an actual sampled answer with periodic full-prefix score audits.
