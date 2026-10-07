# GPU numerical drift: localized to halfway rounding

The bounded localization is complete. On the exact failed 44-token prompt, controlled group-0 captures identify a compute-policy-dependent rounding difference: CPU results follow ties-to-even; GPU-policy results follow ties-away-from-zero at the captured final rounding operation. This explains concrete first-layer projection and rotary-key discrepancies. It does **not** yet prove that correcting rounding will eliminate every full-model score error. The original GPU numerical gate remains failed and unchanged; no GPU candidate is promoted.

## Decisive examples

| Captured site | Identical value before rounding | CPU result | GPU-policy result |
| --- | ---: | ---: | ---: |
| Cold key projection, tokens 12/14, flattened channel 199 | 0.791015625 | 0.7890625 | 0.79296875 |
| Cold final rotary scaled quotient, token 1/head 0/channel 1 | -248.5 | -248 | -249 |
| Warm final rotary scaled quotient, local token 0/head 0/channel 1 | -224.5 | -224 | -225 |

For the cold rotary example, both policies have the same raw sum -3.8828125 and scale 0.015625. Rounding the identical quotient -248.5 differently produces keys -3.875 and -3.890625. For the warm example, equal raw sum -3.5078125 and scale 0.015625 produce keys -3.5 and -3.515625. Thus these discrepancies arise at rounding with identical upstream values; upstream accumulation error is not required for these counterexamples.

Across all captured final-round values, CPU matches NumPy ties-to-even and GPU-policy matches ties-away-from-zero: 8192 cold plus 3072 warm elements per policy. This describes these observed finite arrays, not every possible input or a particular GPU machine instruction. Actual per-operation device placement remains unverified.

## Controls and evidence

| Gate | Result | Evidence |
| --- | --- | --- |
| Reproduce failed prompt | PASS: all 6 earlier hidden metric dictionaries reproduced; all 6 CPU captures exactly match portable reference | [Frozen protocol](replay-protocol.json), [raw replay](prompt0-1/results.json) |
| Separate policy/input/cache | PASS: 60 exact restored-state replays, 20 diagonal output/state controls;4 plain+80 state guarded predictions | [Raw replay](prompt0-1/results.json), [independent audit](independent-audit.json) |
| Minimal four-tap clone | PASS: 42,498 original operations preserved; original endpoints and all 8 states byte-exact in every policy/chunk cell | [Build](key-taps-1/results.json), [eight-call replay](key-taps-replay-1/results.json) |
| Adjacent raw/round taps | PASS: 13 existing values exposed without arithmetic edits; all endpoint/state controls pass; shared four taps exactly reproduce the prior clone | [Build](raw-key-taps-1/results.json), [eight-call replay](raw-key-replay-1/results.json) |
| Actual execution tracing | INCONCLUSIVE: both workers complete and match saved arrays, but owned Core ML tables contain zero events | [Placement report](placement-summary.md) |
| CPU preservation | PASS: both original and promoted source/compiled cohorts, 38 qualified sources and 37 integration source snapshots remain unchanged | [Final preservation](cpu-preservation-final.json) |

The initial four-tap probe bracketed the cold discrepancy between normalized input and rounded projection. The raw probe narrows it further: **raw projection x_9 is exact under both policies**, including the two halfway values above. Warm normalized keys and raw sine/cosine products are also exact; rounding introduces 5 cosine-product and 23 sine-product differences before the final rotary round. The two initially separate locations therefore share a concrete halfway-rounding pattern.

The first layer's rotated key differs at 343 cold and 150 warm elements. This includes propagation and rounding interactions; these counts are not independent causal contributions. Cold key normalization differs at 4 elements after the 2 rounded-projection differences. All groups 0–4 still show policy-dependent output differences on shared input/cache controls. Input and warm-cache effects are recorded separately; nonlinear effects cannot be added as error percentages.

The first earlier hidden tolerance crossing at layer 19 is downstream, not the origin. This bounded run covers embedding and groups 0–4, without recomputing the final score head or claiming a new generation result. The old full-model numerical failure remains the relevant qualification result.

## Limits and next gate

Adding outputs can change compilation. Exact original output/cache controls, plus agreement between two independently instrumented tap sets, support these observations; they do not expose the original uninstrumented hardware instruction stream. The trace attempt could not resolve actual dispatch, so `GPU-policy` throughout means `CPU_AND_GPU`, not proof that every operation executed on GPU.

Next: build an isolated, explicit ties-to-even replacement in a separate candidate; validate directed halfway/non-halfway cases, then original group-0 saved input/cache cases under both policies. Only after that passes should it be tried across the full graph and rerun through the unchanged full-model numerical gates. Preserve the qualified CPU paths. A rounding fix is a targeted hypothesis to test, not a promised complete cure.

The [independent raw-stage audit](independent-raw-key-audit.json) confirms the rounding result. Among identical cross-policy quotients, 320 cold and 134 warm outputs differ, all at exact half-integers. The [reproducible array analysis](rounding-summary.json), generated by [summarize_gpu_rounding.py](../../scripts/summarize_gpu_rounding.py), additionally matches the same policy-specific rounding law at every captured projection, cosine-product, sine-product and final-sum quantization stage. The two cold scale differences have different incoming raw sums; they do not by themselves establish a separate scale-computation defect.

The [next-fix design](rounding-fix-design.md) also records a signed-zero flaw in the earlier ties-even helper, so it must not be reused unchanged. [Next experiment and gates](next-experiment.json). [Handoff checksums](SHA256SUMS), [verification](checksum-verification.json). All arrays, source/protocol snapshots, cloned model inventories, regular process logs and scoped trace artifacts are kept in this evidence folder or its linked separate artifact folder. [Commands](commands.md).
