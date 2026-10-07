# Warm-score branch experiment (05f)

Status: COMPLETE — local performance GO, full-runtime integration unqualified. All 180 calls preserve exact outputs and all eight cache state arrays. Group-call median reductions are 17.76–22.59%; all nine block/prefix cells exceed the predeclared 10% threshold.

| Gate | Result | Evidence |
|---|---|---|
| S1: isolated graph manipulation | PASS: four selectors now use unchanged ordinary scores; 10,038 exclusively dead short-score operations removed; 32,456 other surviving operations, all 68 matmuls, state operations, metadata and weight blobs unchanged | [Build](warm-only-1/results.json), [graph review](build-review.md) |
| S2: exact fresh-state replay | PASS: 12 initial checks, 24 excluded warmups and 144 timed calls all exact for both outputs and eight states at positions 32/128/512; nine invalid-scope cases rejected | [Raw results](benchmark-1/results.json) |
| S3: repeatable local gain | PASS: 9/9 cells clear 10%; exact schedule/decisions independently aggregated; eight summary rejection tests pass | [Frozen protocol](protocol.json), [summary](summary.json), [tests](summary-tests.log) |
| S4: preserve and qualify separately | PASS for preservation; full integration NOT RUN | Baseline source/compiled identities verified before/after in raw results; final handoff preservation shared with GPU probe |

The variant starts from the qualified baseline, retaining the original value products. It does not combine the previous 05e ablation. The MIL condition comes from the prepared query row count: for one query at position ≥32, the source prepares 32 rows, so the short-prefix score result is never selected. The builder replaces exactly four selectors with ordinary-score identities and removes only the newly unreachable pure upstream operations. Native compilation took 1.48 seconds.

The package keeps its original interface; the experimental harness alone enforces one query at position 32–2047. It must not be inserted into generic prefill or short-prefix routing. Full integration would need a state-ownership and routing design followed by the complete numerical/lifecycle/partition/boundary/generation gates.

## Paired prediction timings

Each cell contains eight observations per arm from four ABBA cycles. The order of prefix conditions rotates across three blocks. Every call starts with a new restored native state. Restoration, exact comparisons and file writes are outside the synchronous prediction timer; bridge and output-conversion costs are inside. Initial correctness calls and warmups are excluded. Guard counters match exactly 180 stateful calls and 180 GIL releases; the process exits normally.

| Block | Position | Baseline median (ms) | Variant median (ms) | Reduction |
|---|---:|---:|---:|---:|
| 1 | 32 | 192.95 | 150.41 | 22.05% |
| 1 | 128 | 193.95 | 150.13 | 22.59% |
| 1 | 512 | 194.65 | 151.01 | 22.42% |
| 2 | 32 | 194.06 | 151.12 | 22.13% |
| 2 | 128 | 196.72 | 153.72 | 21.86% |
| 2 | 512 | 193.38 | 152.15 | 21.32% |
| 3 | 32 | 199.35 | 158.94 | 20.27% |
| 3 | 128 | 201.87 | 166.02 | 17.76% |
| 3 | 512 | 207.95 | 161.77 | 22.21% |

This is a local group0 latency improvement on the pinned CPU host. It is not a whole-model tokens/s measurement. Host load changed from 3.03 to 3.73, with AC power and no reported thermal/performance warning. Host activity, caches, fresh-state restoration and native compiler scheduling limit generalization. The qualified CPU baseline remains unchanged.

The next authorized work is the [original-graph GPU probe](../06-gpu/REPORT.md). A later CPU integration investigation remains viable because this experiment passed. [Commands](commands.md), [decision review](decision-review.md).
