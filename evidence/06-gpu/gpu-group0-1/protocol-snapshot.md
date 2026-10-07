# Group 0 CPU/GPU same-graph diagnostic

This is a bounded diagnostic of the original sealed candidate 6 compiled group
0 model. It compares `CPU_ONLY` with `CPU_AND_GPU` on the same compiled graph.
It does not change the runtime loader, establish end-to-end GPU use, or claim a
speedup. The CPU product ablation and any derived artifact are outside this
probe; only `artifacts/05c-compiled/full-cache-1/0.mlmodelc` is selected.

The probe verifies the pinned source package, compiled aggregate, runtime
provenance, and loader provenance before loading. It loads the compiled CPU
embedding and the same group 0 artifact under both compute policies. CPU
embedding creates actual cold-prefill group inputs for the first 3 and 32 IDs
of the pinned generation stream. Cold state starts at zeros with the exact
sealed eight-state schema. Warm inputs and starting states come from the
passing 05e captures at prefixes 32, 128, and 512. Each CPU warm replay must
exactly reproduce its sealed output and all eight post-states before GPU
replays are accepted.

The call plan is two embedding calls, eight CPU group calls (two cold
baselines and two warm replays at each of three prefixes), and ten GPU group
calls (two replays for each of two cold cases and three warm cases): 18 state
calls total. Every GPU call is retained even after a mismatch. The primary
same-graph device-parity gate is strict exact equality of both outputs and all
eight states, including exact shape and dtype, with exact state restoration.
The frozen 0.01 relative-RMSE and 0.995 cosine cutoffs are recorded as
diagnostic measures only; they are not substitutes for exact parity or global
qualification gates. Non-finite values are counted and retained, while
undefined numeric summaries are stored as null.

State arrays read through the Core ML state API are exposed as float32 here;
the source model's state representation is FP16. Exact comparison therefore
describes the exposed arrays for this same-graph device comparison, not a claim
about source-storage bit patterns. The compute plan is an anticipated placement
report for `CPU_AND_GPU`, not proof of where operations ran at prediction time.

The attempt writes each input, starting state, output, and post-state under a
unique attempt directory, records per-output and per-state differences, and
checks source, compiled artifact, runtime, prediction guards, and pinned
protocol/fixture hashes again at the end. A completed exactness no-go is a
valid diagnostic result with no speedup claim. A failed CPU control, identity
check, preservation check, or guard check invalidates the attempt.

Pinned inputs:

- Source candidate aggregate: `5a3e4c7a571e4fc3532377d9b9bb0da023f499c10709216fee7e1001bf4ad2c6`.
- Compiled aggregate: `a838dd36e8f87c6b9fc0dd7865d0912651c6ab3399fa299d10ff12d788bcb031`.
- 05e protocol SHA-256: `9d9e0656b224975d27551ca489f3448c69d62bc7401c4cef48d232f7b50a6606`.
- Generation comparison fixture SHA-256: `1ee6553885123a57b8fb40b89c292eea201e148e6f3bd4d9b4cba1c057f9f3eb`.

No inference is launched by this protocol document or its static review.
