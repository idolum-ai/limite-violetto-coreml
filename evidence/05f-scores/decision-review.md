# Independent 05f warm-score decision review

I recomputed all nine block/prefix median reductions from the raw A/B rows in `benchmark-1/results.json`. Each cell contains eight baseline and eight variant calls. All 12 correctness, 24 excluded warmup, and 144 timed calls report exact outputs and all eight states: **180/180 exact**. The nine median reductions span **17.76%–22.59%**, and **9/9 exceed the predeclared 10% threshold**. The recorded local performance-go decision is therefore supported by the raw benchmark.

This is a group-0, warm single-query result against a copied baseline-derived score variant. It does not establish full-model throughput, general input correctness, GPU behavior, or authorization to replace the qualified runtime. No inference or compilation was run for this review.
