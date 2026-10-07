# Tiny-prefix CPU score arithmetic: bounded operator result

The opt-in [score module](../../coreml_port/tiny_native_scores.py) is frozen at
SHA-256 `c1fd3339ce3ac7b23cf0117dfb466b25c2c43c036a87616eb8656787fa5858ff`.
It reproduces native Torch CPU raw dots and scaled scores for square logical
query/key lengths 1–15 on the measured macOS 26.5.2 arm64 host with Torch
2.7.0 and Accelerate. The [independent Torch holdout](module-torch/summary.json)
passed all 45 length/seed cases across every head, query row, and active key.
The [isolated dynamic Core ML operator](dynamic-operator-1/summary.json) was
compiled with FLOAT32 precision and run on CPU_ONLY with the required
prediction guard. It matched the Torch module across all 2048 returned score
slots at Q=1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,32; every active score also
matched native Torch. This is operator evidence, not a composed prefix or
full-model gate.

The [direct SGEMM reproduction](direct-sgemm/summary.json) calls Accelerate
`sgemm_` with the arguments selected by the retained Torch `addmm` source:
column-major `T,N`, `m=K`, `n=Q`, `k=128`, interleaved-head leading dimensions
1280, `ldc=K`, and alpha/beta 1/0. It matched every saved raw dot and scaled
score at square lengths 2,3,7,15 across three seeds. The [fresh rectangular
matrix](rectangular-sgemm/summary.json) matched all 61 cases above Torch's
`<400` naive-kernel cutoff, including Q3/K2048. Five smaller cases differ,
as expected because Torch selects its naive CPU kernel there. This confirms
the invoked BLAS backend and operand geometry, without identifying Apple's
private internal microkernel.

[Cancellation-triplet probes](q3-tail-tree-key0/summary.json) recovered a
binary FP32 addition tree for each measured output position. Their helper's
[self-test](sum-topology-self-test.json) reconstructed sequential, reverse,
four cyclic-lane counts, and a random tree, and rejected an FP64 sum. Q3/K3
tail row 2 uses four cyclic lanes. Key columns 0–1 fold the lanes as
`((0+1)+(2+3))`; key column 2 folds them as `((0+2)+(1+3))`. Each recovered
tree passed 1,000 unseen cancellation triplets, sign reversal, a second
amplitude, and three seeds of dense BF16-valued dots at its coordinate. The
[Q7 tail](tree-q7-k7-r4-c0/summary.json) and [Q8 tile](tree-q8-k8-r0-c0/summary.json)
yield two additional trees. At Q3/K2048, measured tail columns 0 and 2 use
the sequential tree, showing that logical key width changes the selected
reduction. For the tested finite, normal-range BF16 operands, each scalar
product is exactly representable in FP32; this claim does not cover arbitrary
FP32 operands, underflow, or overflow.

The [square candidate matrix](square-candidate-matrix/summary.json) assigns
those four trees plus sequential accumulation by `(Q, query row, key column)`.
Three fresh seeds made every coordinate at Q=1–15 compatible with at least
one tree across all ten heads. A strict one-tree assignment remained at every
coordinate. The [frozen-map holdout](square-holdout/summary.json) then passed
45 independent cases, all raw dots and scaled scores bitwise exact. The map
follows tiled bands: Q3–5 use the two four-lane folds and sequential borders;
Q6–7 use sequential first four rows and the Q7 tail tree thereafter; Q8–14
combine an eight-column Q8 tile, sequential off-diagonal bands, and a Q7
style lower-right tail; Q1–2 and Q15 are sequential. The literal constants
and placement in the module express this observed host-specific geometry.

The frozen map's exact generating script is retained as
[source-snapshot.py](square-candidate-matrix/source-snapshot.py), whose hash
matches that summary. Its [provenance note](square-candidate-matrix/provenance.md)
records discovery reruns. The runtime and existing model packages remain unchanged. Explicit converter
and builder flags integrate the frozen module into candidate 6; the completed
preflights and remaining full-package gates are recorded below.

The frozen native-score module (SHA256 `c1fd3339ce3ac7b23cf0117dfb466b25c2c43c036a87616eb8656787fa5858ff`) passes45 independent Torch operator cases acrossQ1..15 with exact active raw/scaled scores. A dynamic CPU Core ML package passes the tested tiny shapes and ordinaryQ16/Q32 paths across all2048 score slots; Q10 is being added as supplementary coverage. The full63-prefix Torch audit and actual integrated converter group8/9 replays are running. [Torch module](module-torch/summary.json), [dynamic Core ML operator](dynamic-operator-1/summary.json), [whole-model prefix audit](native-prefix-preflight/results.json).

The candidate aggregator now has an explicit `paired-native-tiny` policy requiring native helper source identity, both builder flags and the matching selections in all12 group conversion commands. Eleven policy tests pass; this is provenance validation, not model fidelity. [Policy audit](policy-audit/verification.json).

## Completed preflights and candidate 6 build

The whole-model Torch preflight passed **63/63 prefixes**, with hidden states and scores bitwise exact in every comparison (157.38 seconds). This spans four fixtures and prefixes 1–16, with the 15-token fixture ending at 15. [Summary](native-prefix-preflight/summary.json), [full results and source hashes](native-prefix-preflight/results.json).

The saved Core ML operator additionally passed Q10 with a fresh seed, completing Q1–16 and Q32 coverage. Actual converter group 8 and group 9 packages reproduced their saved native boundaries bitwise; group 9 restored the captured pre-call cache state. [Q10](q10-supplement/summary.json), [integrated replay evidence](integrated-replays.md).

| Gate | Status | Evidence |
|---|---|---|
| T1 Native backend binding | PASS, 61 rectangular cases | [Results](rectangular-sgemm/summary.json) |
| T2 Reduction map and independent holdout | PASS, 45 fresh cases | [Results](square-holdout/summary.json) |
| T3 Torch and Core ML operator | PASS for bounded shapes | [Torch](module-torch/summary.json), [Core ML](dynamic-operator-1/summary.json), [Q10](q10-supplement/summary.json) |
| T4 Composed prefix and integrated groups | PASS, 63 prefixes and both saved groups | [Prefixes](native-prefix-preflight/summary.json), [groups](integrated-replays.md) |
| T5 Full candidate qualification | RUNNING: candidate 6 conversion | [Build](../04d-cpu/full-candidate-6/build.json) |

Candidate 6 selects paired value products and the frozen native tiny-score module. Conversion sources are snapshotted before the build; all numerical acceptance thresholds remain unchanged. Full Core ML short, partition, boundary and generation gates must pass before qualification. These preflights alone do not qualify the full package.

T5 update: candidate6 passes full short54/54, partition97/97, boundary6/6 and lifecycle/repeat gates, with hidden states exact throughout. [Full-model summary](../04d-cpu/full-candidate-6/summary.json). Generation remains running; T5 remains PARTIAL.

## Final T5 result: PASS

Candidate 6 passes every complete numerical, lifecycle and generation gate. The 674-token response matches the qualified CPU reference exactly, and all seven generation score audits pass. [Aggregate qualification](../04d-cpu/candidate6-milestone.json), [current report](../04d-cpu/REPORT.md). This is the bounded pinned-host CPU result; accelerated devices and original NVIDIA fidelity remain unqualified.
