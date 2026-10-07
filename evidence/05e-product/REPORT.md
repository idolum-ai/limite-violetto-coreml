# Single-token attention-value experiment (05e)

Status: COMPLETE — performance gate NO-GO. Removing unused value products preserved exact results, but observed median group-call reductions of 4.82–10.06% met the predeclared 10% threshold in only one of nine block/prefix cells. The experimental artifact is retained for research and is not integrated into the qualified runtime.

## Validation gates

| Gate | Result | Evidence |
|---|---|---|
| E1: narrow graph change | PASS: four concat rewires, 60 unreferenced matmuls removed (68→8), 42,434 other surviving operations byte-identical; descriptions, states and weights unchanged; native compilation succeeds | [Build](repeat-first-2/results.json), [review](build-review.md) |
| E2: exact fresh-state replay | PASS: all 180 calls match both outputs and all eight KV state arrays; 12 initial correctness calls across positions 32/128/512 precede timing; nine invalid-scope cases rejected | [Raw results](benchmark-1/results.json), [position 32](../05d-decode/group0-warm-capture/results.json), [position 128](capture-p128/results.json), [position 512](capture-p512/results.json) |
| E3: repeatable useful speedup | FAIL: only 1/9 cells reaches 10%; observed reductions 4.82–10.06%, median cell reduction 7.38% | [Frozen protocol](protocol.json), [independent summary](summary.json), [benchmark review](benchmark-review.md) |
| E4: preservation and adoption decision | PASS: qualified source/cache identities and 30 source/runtime/validator files preserved; variant not adopted | [Preservation](baseline-preservation.json), [process exits](process-exits.json) |

The experiment completed normally. E3 is a failed acceptance gate, not a crashed or incomplete benchmark. No numerical tolerance was widened.

## What changed

A copy of the qualified group0 MIL graph repeats the original first two-row value product in all 16 concat slots, then removes the 60 matmuls whose outputs have no remaining consumers. The concat geometry remains unchanged; the first returned query row retains the original operand geometry. All other operation bytes and weight blobs are unchanged. Dead slice/constant ancestors remain for the native compiler to handle.

This artifact is valid only through the experiment harness, which enforces one real query at absolute position 32–2047 before creating state or calling Core ML. The package retains its original broad interface to avoid introducing shape specialization as a second change. It must not enter generic prefill or generation routing.

The first build attempt failed before saving or compiling: its checker assumed every operation had an output, but state-write operations have none. The corrected checker retains and compares those operations in their original order. The [failed result](repeat-first-1/results.json), [log](repeat-first-1.log) and source snapshot are preserved; the second build compiled in 1.85 seconds.

## Exactness and timing

The two new fixture captures independently verify full-model scores against the pinned historical reference, then replay each group call twice with exact outputs and states. The existing position-32 capture is checked against the completed 05d checksum seal. All fixtures, build results and benchmark source are pinned in the protocol before variant execution.

Every baseline and variant call receives a fresh native state populated with identical captured values. The initial 12 correctness calls pass before 24 excluded warmups. Three blocks rotate prefix order, each using four ABBA cycles per prefix, for 144 timed predictions. State creation/restoration and exact comparisons are outside the synchronous prediction timer. The measured boundary includes bridge and output-conversion cost. The prediction guard records exactly 180 stateful calls, 180 GIL releases and zero stateless calls; the process exits normally.

| Block | Position | Baseline median (ms) | Variant median (ms) | Reduction | ≥10% gate |
|---|---:|---:|---:|---:|---|
| 1 | 32 | 189.69 | 174.68 | 7.92% | FAIL |
| 1 | 128 | 228.42 | 214.60 | 6.05% | FAIL |
| 1 | 512 | 212.08 | 195.34 | 7.89% | FAIL |
| 2 | 32 | 209.33 | 189.74 | 9.36% | FAIL |
| 2 | 128 | 220.44 | 198.27 | 10.06% | PASS |
| 2 | 512 | 215.15 | 199.63 | 7.22% | FAIL |
| 3 | 32 | 221.72 | 205.35 | 7.38% | FAIL |
| 3 | 128 | 223.59 | 209.23 | 6.42% | FAIL |
| 3 | 512 | 225.92 | 215.04 | 4.82% | FAIL |

All nine cells favor the variant, but eight miss the predeclared threshold. This is a modest local group0 observation, not a measured token-generation speedup. Each cell contains eight timings per arm. The host was on AC power with reported load 5.15→4.96 and no thermal/performance warning. Other host activity and caches were uncontrolled. Fresh-state restoration affects cache conditions; ABBA and rotated order reduce, but do not eliminate, timing bias. No other inference experiment ran concurrently.

The result supports keeping this narrow variant as evidence, while leaving the qualified runtime unchanged. It does not establish how much time individual internal operators consume, whether the native compiler had already optimized some discarded work, or how savings would scale across all 12 groups.

## Next bounded move

Inspect and separately test the warm-score path: for warm positions the padded query has 32 rows, so the short-prefix score branch is not selected. A baseline-derived group0 ablation could remove only that branch while preserving the ordinary score expression and the original value products. This is a new hypothesis, not a promised speedup; it must use the same structural, exact-state and paired-timing gates. Do not combine it with this variant before measuring it independently. [Next experiment](next-experiment.json), [independent decision review](decision-review.md).

[Commands](commands.md), [summary tests](summary-tests.log), [checksums](SHA256SUMS), [checksum verification](checksum-verification.json). Original NVIDIA-runtime fidelity, GPU/ANE support, clean-machine installation and broad task accuracy remain outside this experiment.
