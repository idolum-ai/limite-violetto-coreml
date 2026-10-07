# Candidate-4 three-token score divergence

The frozen three-token fixture `[87555, 130664, 114854]` first differs in
candidate-4 group 9 (layers 36–39): groups 0–8 are bitwise exact at their
boundaries, while group 9 differs in 449 of 3,840 BF16 hidden elements. The
corrected Torch graph reproduces the native group-9 output exactly on the
saved input and cache state. See the [independent group locator](candidate4-tiny3-firstgroup/results.json)
and [group-9 Q-site replay](candidate4-tiny3-group9-stages/results.json).

The first differing Q site is `round_81`, layer 38's weighted attention
product, with one element `[0,2,8,38]` at `-0.185546875` in Core ML versus
`-0.1845703125` in Torch. All 81 earlier Q sites agree. Given identical
logits, isolated Core ML source softmax weights agree bitwise; given identical
weights and values, its raw and Q-rounded value products agree bitwise. Given
identical Q and K, the Core ML score GEMM differs at 53 FP32 elements (maximum
`2.384185791015625e-6`), and passing those scores through the Torch softmax
and value product reproduces exactly the one Q failure. The
[operator isolation](candidate4-tiny3-layer38-isolate/summary.json) retains
its source tensors and Core ML outputs as NPZ files.

Cropping the key operand from 2,048 to the three active keys does not resolve
this fixture: the Core ML score still differs at 44 of 90 active elements and
the same Q element fails. See the [active-key ablation](candidate4-tiny3-layer38-active-key/summary.json).
Explicit sequential FP32 accumulation of 128 BF16-exact products matches all
90 native Torch active scores bitwise, both with K3 and K2048, and Core ML
preserves that expression in an isolated operator. The resulting rounded
attention tensor is exact in all 3,840 elements. See the
[sequential-dot probe](candidate4-tiny3-layer38-sequential/summary.json).

Replacing group 9's scores in all four layers with the sequential expression
makes its saved tiny3 replay exact in both Torch and Core ML. Restricting the
sequential calculation to the first 16 key slots and zero-padding the
remaining scores before the existing causal mask is also exact on this
capture. The [fixed-shape composed probe](candidate4-tiny3-group9-shortkey/summary.json)
and [dynamic-shape composed probe](candidate4-tiny3-group9-dynamic-scores/summary.json)
each report zero group-output differences. The tested dynamic expression is
in [`coreml_port/tiny_scores.py`](../../coreml_port/tiny_scores.py): for fewer
than 16 prepared query rows, it uses sequential first-16-key scores; otherwise
it selects the existing raw-layout einsum scores. A single Core ML package
with query RangeDim 1–32 matches the three-row native active scores and the
32-row ordinary scores bitwise in the
[branch operator test](tiny-scores-dynamic-operator/summary.json).

The all-12-group Torch tiny3 boundary preflight shows groups 0–7 and 9–11
exact with this policy. Group 8 differs in 789 elements, exactly as it does
under candidate-4's original score policy on the same native boundary input;
see the [short-score run](candidate4-tiny3-torch-short-scores/summary.json)
and [baseline run](candidate4-tiny3-torch-baseline/summary.json). Candidate-4
Core ML group 8 was independently exact. This source-graph distinction means
that a full composed Core ML gate is still necessary. Groups 10 and 11 of the
ordinary candidate are independently exact on native inputs, as recorded in
the [remaining-groups replay](candidate4-tiny3-remaining-groups/results.json).

This is a causal correction for the measured three-token group-9 failure,
not proof of general score-GEMM identity. The proposed dynamic score policy
has not yet been combined with the separate long-capacity paired-value-product
correction or run through a complete candidate gate.
