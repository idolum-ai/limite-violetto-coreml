# Downstream interval replay design

The first controlled normalization probe established that the observed layer1 pre-attention reciprocal error does not survive its final quantized output on the warm fixture. Its denominator exceeds the existing helper domain, so no correction is inserted. Preserve this negative result and the frozen first attempt.

Use the same 06f baseline, original prior candidate expectations, cold32/warm12 input archives and restored CPU states as replay-1. Add only existing fixed-shape boundary outputs spanning layer1 attention/post-attention normalization/MLP to layer2 pre-attention normalization and key projection. Exact tap names, shapes and dependency checks are frozen in interval-design.md and the builder preflight before compilation. Avoid exporting the prior failing data-dependent scores/weights outputs.

Eight guarded stateful calls: original and tapped models under CPU_ONLY and CPU_AND_GPU on both chunks. Each original endpoint and all eight states must reproduce the saved06f values. Each tapped endpoint and all states must match its same-policy original. Verify source/package/compiled/input/archive identities and guard count8 before interpretation. Independent analysis checks raw controls and every exposed stage, reporting the first differing boundary and its preceding exact boundary without assuming an underlying operation is causal. Compute policy does not establish device placement.

Freeze interval-replay-1-protocol.json with worker/build/baseline/prior/fixture/design/helper identities and all input/expected archives before native execution. This is a separate diagnostic with no mathematical rewrite and no full-model qualification.
