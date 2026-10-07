# Candidate 6 full-model GPU-policy short screen

Status: **completed numerical NO-GO**. The isolated `CPU_AND_GPU` run completed all 54 comparisons and 17/17 lifecycle checks on the original sealed source/cache. Lifecycle passed; numerical fidelity failed. The unchanged score threshold is maximum absolute error <= 0.04, with the original top-1/reference-margin rule; hidden thresholds remain relative RMSE <= 0.01 and cosine >= 0.995. No thresholds changed. [Raw result](short-1/results.json), [independent audit](independent-summary.json), [protocol](PROTOCOL.md).

All **15/15 score comparisons fail** the 0.04 maximum-absolute-error condition, although top-1 matches in 15/15 and top-5 overlap is 4 at worst. The worst score error is 0.315598 at `prompt1/teacher1`; its relative RMSE is 0.028632.

Of 39 hidden comparisons, **21 pass and 18 fail** the frozen combined RMSE/cosine screen. Cosine remains above the threshold; failures are relative-RMSE crossings. For each prompt, the first measured hidden threshold crossing is `layer_19`: prompt 0 RMSE 0.010770, prompt 1 0.011179, prompt 2 0.010969. Earlier captured hidden checkpoints pass the configured limit. The worst hidden relative RMSE is 0.024932 at `prompt2/final_hidden`. These are threshold crossing locations, not proof of where numerical error first arose or its cause.

All **17/17 lifecycle checks pass**. The runner loaded 14 compiled model packages. Source candidate, compiled cache, runtime provenance, recorded code inputs, script and protocol passed preservation checks. Prediction guard counters advanced by 52 stateless calls, 360 stateful calls and 412 GIL releases; cleanup guard reported zero background calls and zero shutdown skips.

The run used the sealed source identity `5a3e4c7a571e4fc3532377d9b9bb0da023f499c10709216fee7e1001bf4ad2c6` and compiled aggregate `a838dd36e8f87c6b9fc0dd7865d0912651c6ab3399fa299d10ff12d788bcb031`. `CPU_AND_GPU` permits GPU execution, but actual device placement remains **unverified**. This short gate does not qualify full generation, partitions, boundaries, or performance. Since the original numerical gate failed, those broader GPU runs are not justified by this attempt; CPU fidelity remains the priority.

A minimal next diagnostic experiment is a group-by-group CPU/GPU replay of frozen prompt 0 (44 tokens) through group 4, saving each group's input, all eight starting/ending states, outputs, and hidden captures. The existing group 0 probe is useful context, but its fixture inputs and starting states differ from this prompt and cannot establish the causal source of this run's drift.

The [next diagnostic plan](next-experiment.json) is recorded but has not run. The separate [CPU integration](../05g-integration/REPORT.md) completed qualification and promotion.
