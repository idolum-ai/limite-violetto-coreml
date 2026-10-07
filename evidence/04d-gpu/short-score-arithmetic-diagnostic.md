# Short-prefix score arithmetic: measured failure and negative controls

Candidate 4's frozen three-token fixture `[87555, 130664, 114854]` first
diverges in group 9. Its first differing rounded site is layer 38's attention
value product, one BF16 element. With identical inputs, Core ML's source
softmax and value product are exact, while its query-key score GEMM differs in
53 FP32 elements. Feeding only those Core ML scores through Torch's softmax and
value product reproduces the one BF16 error. This establishes the **causal
operator on that capture**, not a general score-reduction law. See the
[group locator](candidate4-tiny3-firstgroup/results.json),
[rounding sites](candidate4-tiny3-group9-stages/results.json), and
[controlled operator probe](candidate4-tiny3-layer38-isolate/summary.json).

The pinned Torch CPU path uses per-batch `addmm`/`sgemm` for a 3×3 score GEMM
with contraction width 128: its 1,152 scalar products exceed the `<400`
naive-kernel cutoff in the retained
[Torch source](../../sources/torch-2.7-linear/LinearAlgebra.cpp). The installed
Torch 2.7.0 build reports `BLAS_INFO=accelerate`, `USE_MKL=OFF`, and
`USE_EIGEN_FOR_BLAS=ON` in the [captured build config](accelerate-config.txt).
The observed FP32 score differences arise from reduction arithmetic/geometry;
BF16-valued operands make individual products exactly representable in FP32.
The exact internal Accelerate tail kernel has **not** been established.

Explicit sequential accumulation of 128 products happened to match all 90
native scores at layer 38 and produced an exact compiled group-9 replay, even
after computing only the first 16 active key slots. A dynamic-shape package
also passed q3 and q32 operator checks. That result is local:
[sequential operator](candidate4-tiny3-layer38-sequential/summary.json),
[group-9 replay](candidate4-tiny3-group9-dynamic-scores/summary.json), and
[dynamic operator](tiny-scores-dynamic-operator/summary.json). On the same
fixture, sequential scores cause a layer-32 BF16 attention error, and a
compiled group-8 replay differs from native in 789 hidden elements. The
[tiny-score-only ablation](candidate5-tiny3-group8-scores-only/summary.json)
establishes this regression; adding paired value products does not remove it
([combined ablation](candidate5-tiny3-group8-combined/summary.json)). Cropping
the score operand to three active keys also retains the layer-38 error
([active-key ablation](candidate4-tiny3-layer38-active-key/summary.json)).

Independent deterministic BF16-valued q/k probes use the source raw layout
`[1,time,10,128]`, with two K heads repeated fivefold, and native Torch
`einsum('bthd,bshd->bhts', q, k)`. Across three seeds each, sequential scores
are bitwise exact for square query/key lengths 1, 2, and 15, but differ at
45/270 scores for length 3 and 327/1,470 for length 7. The length-3
differences occur **only at query row 2**; length-7 differences occur **only
at query rows 4–6**, with no corresponding concentration in key columns.
These operands, native dots, strides, and reduction comparisons are retained
in the [fresh reduction matrix](bf16-dot-reduction-matrix/summary.json) and
[NPZ capture](bf16-dot-reduction-matrix/operands-and-native.npz). The two
measured model layers also disagree on which simple reduction matches their
scores ([captured-layer comparison](tiny3-dot-order-contrast/summary.json)).

Zero-padding only query rows to 4, 8, or 16 reproduces the sequential-like
difference count on the original active rows; padding keys changes scores
more broadly. An FP64 sum rounded to FP32 before FP32 scaling differs at
129 length-3 scores and 679 length-7 scores across the fresh seeds. Restricting
FP64 to the odd-query tail still differs at 33/90 and 257/630 scores,
respectively. The [shape ablation](bf16-dot-shape-ablation/summary.json)
records row/column counts and coordinates. A search of 175 contiguous-block
sum variants at each length found no exact tail formula
([block search](bf16-tail-block-search/summary.json)). This suggests a
query-row tile/tail kernel in Accelerate; that attribution remains an
inference from numerical behavior, not a disassembly or source proof.

Empirical two-lane short scores with paired value products make the saved
group-8 and group-9 Core ML replays exact
([group 8](candidate5-tiny3-group8-two-lane/summary.json),
[group 9](candidate5-tiny3-group9-two-lane/summary.json)). Yet a native
boundary preflight shows a 480-element group-2 hidden mismatch
([all groups](candidate5-tiny3-torch-two-lane/summary.json)), and the broader
full-Torch prefix gate passed only 10 of 63 tested prefixes, with 53 failures
and maximum score error `1.5667943954`
([two-lane prefix gate](../04d-cpu/two-lane-prefix-preflight/results.json)). Four-lane
forward and adjacent folds make all 12 saved tiny3 group boundaries exact
([forward](candidate5-tiny3-torch-four-lane-forward/summary.json),
[adjacent](candidate5-tiny3-torch-four-lane-adjacent/summary.json)), but the
broader adjacent-fold prefix gate passes only 20 of 63 prefixes and fails 43,
with maximum score error `1.0187029838562012` and first failure at
`tiny_33` length 6
([prefix results](four-lane-prefix-preflight/results.json)). These are
**negative gate results**; local BF16 success is not a reason to integrate
either arithmetic policy.

The next source-faithful route would be to characterize the installed
Accelerate `sgemm` query-row tail and validate an explicit implementation
against independent BF16 inputs across the actual Q/K shapes, then run the
complete Torch and Core ML gates before integration. A short-prefix native
CPU fallback is a separate runtime design option if that kernel cannot be
reproduced in Core ML; it would also need a defined cache handoff. The paired
value-product correction for long capacity is independent of this unresolved
short-prefix score issue.
