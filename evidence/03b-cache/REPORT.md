# Cache follow-up — CPU reference qualified

Completed 2026-09-22. **The CPU cache now passes the unchanged Step 3 numerical criteria.** All 467 original comparisons and 137 additional matched-prefix comparisons have zero observed difference. Metal remains unqualified, and original NVIDIA-runtime fidelity remains unverified. No complete Core ML conversion was attempted in this follow-up.

The [live experiment history](REPORT-live-history.md), [operator trail](../../OPERATOR-TRAIL.md), and failed runs remain intact. [Summary](summary.json) contains the current machine-readable result.

## Final behavior and evidence

| Check | Result | Evidence |
|---|---|---|
| Original numerical suite: all layers, mixers, scores, cache/window boundary and teacher forcing | **467/467 PASS; maximum absolute difference 0** | [CPU sixteen-row suite](cpu-sixteen/results.json) |
| Original behavioral/summary checks | **35/35 PASS** | [Suite log](cpu-sixteen.log) |
| Additional tiny/irregular prefixes | **12 cases, 137 matched-prefix comparisons; hidden and score differences 0** | [Extended stress](cpu-partitions-v3.json) |
| Warmup ownership, reset, interruption and disposal | **8/8 PASS** | [Lifecycle checks](warmup-lifecycle.json) |
| Actual sampled mathematical answer | **Complete, correct 391; 632 generated tokens** | [Generation](cpu-generation.json) |
| Score audits during that answer | **6/6 exact matches to full-prefix scores** | [Audit log](cpu-generation.log) |
| Metal cache | **FAIL**, including the query-padding ablation | [Metal results](mps-window/results.json) |
| Original NVIDIA runtime | **NOT EXECUTED** | [Baseline requirements](../01-reference/BASELINE-CAPTURE.md) |

The original [acceptance file](../03-portable/ACCEPTANCE.json) is unchanged: SHA-256 `cb4d315266126df4a98194f73f8c10671310717c06ebfd191d3dac2334bf4233`. Score tolerance remains 0.04, hidden relative RMSE 0.01, minimum cosine 0.995. No failed historical run was reclassified as passing.

## What changed

[portable/model.py](../../portable/model.py) now uses two bounded CPU measures:

1. **At least sixteen attention query rows**, clamped to the available key count. Extra rows are zero-filled dummy queries with valid masks; their outputs are discarded. Real queries retain the same positions, keys, values, causal/local masks, dtypes, and BF16 rounding points. Full-prefix calls retain their original shape. The extra work is limited to attention; it does not repeat the embedding, projection, mixer or MLP calculations for those dummy rows.
2. **Refresh tiny initial prefixes while the cache contains fewer than sixteen tokens.** The cache temporarily owns a copy of at most fifteen token IDs. On the next call, it rebuilds that small prefix into fresh KV state, returns/captures only newly requested rows, and transfers the completed state. Once the prefix reaches sixteen tokens, the temporary IDs are discarded and ordinary cached decoding continues. Failure invalidates the original cache; reset disposes of both IDs and KV state.

Normal templated prompts already exceed sixteen tokens and skip the refresh. CPU is the only backend with this default policy. Metal retains the original unqualified reduction path. The loader exposes `attention_min_query_rows=0` for reproducing the native CPU ablation, not as the qualified generation default.

Checkpoint tensors, scalar folding, RMS epsilon, output head, source files, tokenizer assets, and published EOS default were not changed. The final source snapshot is under [final-code](../../artifacts/03b-cache/final-code/).

## Why this was necessary

The initial problem was numerical shape dependence, not evidence of an incorrect causal-window inequality. Padding attention queries to the full prefix made the three short CPU cases and the 1,026-token window case exactly match the upstream source. However, that strategy would make decode attention quadratic in prefix length, so smaller fixed row counts were evaluated.

Four rows failed two initial prompts. Eight rows passed the original suite, but a new 16-to-17-token case exposed another reduction-shape discrepancy. Sixteen and thirty-two rows matched that transition exactly; sixteen was selected and retested through the complete suite. The final policy bounds the query dimension for single-token decode at sixteen instead of the whole prefix.

A second issue occurs before that stable shape regime: the **unmodified upstream calculation itself** changes some early hidden states when a tiny prefix is extended. On a deterministic valid-token sequence, prefix lengths 3/4/7/8/9 differed from the same causal positions evaluated with a 127-token prefix; tested lengths 16/32 matched. This is why preserving the early rounded KV states can disagree with a later full-prefix calculation. The bounded refresh corrects those states.

The first extra stress test compared all historical returned rows to a later final prefix and failed. Its [original results](cpu-partitions.json) remain. The follow-up compares each newly returned chunk with the source at **the same available prefix**, and also retains the future-prefix discrepancy as a diagnostic. This avoids attributing the source's own prefix-shape variation to the cache. The original Step 3 gate cases and numerical limits were unchanged.

These are empirical findings on this **Apple M4 CPU, Torch 2.7.0, four CPU threads**. Revalidate the numerical policy if the device, library or relevant implementation changes. They are not a universal statement about PyTorch kernels.

## Rejected and retained alternatives

| Experiment | Finding | Evidence |
|---|---|---|
| Full-prefix-size query padding, CPU | Exact on three prompts and window input; unnecessary quadratic attention cost | [Short](cpu1/results.json), [window](cpu-window/results.json) |
| FP64 attention | Cache internally consistent on initial prompts, but source outputs changed; not adopted | [CPU ablations](cpu1/results.json) |
| Four/eight/sixteen/thirty-two query rows | Four insufficient; eight insufficient for the added 17-token transition; sixteen selected | [Row sweep](cpu-small-rows/results.json), [17-token diagnosis](tiny17-diagnosis.json) |
| Query padding on Metal | Still exceeds original thresholds | [Metal ablation](mps-window/results.json) |
| Tiny-prefix source invariance | Source itself changes below the tested stable regime | [Prefix probe](prefix-shape-probe.json) |
| Eight rows plus tiny-prefix refresh | Original suite passes; added 17-token case fails | [Suite](cpu-final/results.json), [stress v2](cpu-partitions-v2.json) |

The old numerical failures remain useful regression evidence. No claim is made that every possible prefix or partition has been exhaustively tested.

## EOS and actual generation

The already-pinned value-model README explicitly identifies completed rollouts with **endoftext/151643** and distinguishes **im_end/151645** as the prompt-turn delimiter. This relevant prose was missed in the earlier audit. [Source evidence](eos-source-evidence.json) records its exact location and hash. It supports the observed local actor emissions, but it does not execute the original Violetto serving runtime or eliminate the conflicting actor configuration.

The published default remains EOS151645. The successful fresh run used the explicit `--stop-at-tokenizer-eos` policy, with recorded stop set `[151643,151645]`, published sampling settings (temperature 0.6, top-p 0.95), and seed 0. It produced a correct boxed 391 and stopped after 632 tokens. Six periodic/terminal head-score comparisons were exact; this does not assert that every generated step was independently audited. Timing includes those audits and overlapping investigation work, so it is not a benchmark.

G3.4 remains partial with respect to the conflicting published stop metadata; loop mechanics and the explicitly chosen local policy pass. G3.2 remains blocked for independent original-runtime fidelity. **G3.3 now passes for the qualified CPU implementation; it still fails for Metal.**

## Reproduce

From the project root, use a fresh attempt name for the full suite:

```sh
.venv/bin/python scripts/validate_portable.py --device cpu --phase 03b-cache --attempt rerun1 --max-new-tokens 1
.venv/bin/python -m portable.generate 'Compute 17 * 23.' --device cpu --max-new-tokens 1024 --sample --seed 0 --stop-at-tokenizer-eos
```

The suite's one-token generation calls intentionally exercise length stopping; the separate 632-token run validates a completed response. Check `all_local_checks_passed` in the JSON: the diagnostic suite records failures without treating a normal process exit as acceptance. [Commands](commands.md) lists the supporting experiments. [SHA256SUMS](SHA256SUMS) covers the final project artifacts; earlier checksum lists are retained.

The next planned work is full-model Core ML conversion against this qualified CPU reference. Preserve the same numerical gates and separately measure conversion-induced precision changes. Metal/PyTorch agreement and original NVIDIA-runtime fidelity remain open.
