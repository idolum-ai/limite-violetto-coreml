# Candidate 6 full-model short GPU-policy gate

This attempt runs the existing short numerical and request-lifecycle suite on
the original sealed candidate 6 source packages and full-cache-1 compiled
artifacts, with Core ML compute units set to `CPU_AND_GPU`. It is an isolated
diagnostic adapter declared in `scripts/validate_full_gpu.py`. It does not
change `coreml_port/loader_selection.py`, the qualified CPU runtime, score or
hidden-state limits, or model math. `CPU_AND_GPU` permits GPU use; actual
operation placement is not traced and remains unverified.

Before constructors, the runner checks the source candidate aggregate, all 14
compiled package bytes and source binding, compiled manifest, current runtime
and guard provenance, current compiled-loader source hashes against the
passing CPU milestone, and the pinned validation input and acceptance-file
SHA-256 values. It installs the prediction and cleanup guards before loading
models. Its experimental pipeline duplicates only the compiled model-loading
constructor with `CPU_AND_GPU`; request ownership, input validation, chunking,
normalization, state handling, capture behavior, and tiny-prefix replay are
inherited unchanged from `TinyPrefixPipeline` / `Pipeline`.

The numerical screen is exactly 54 comparisons across the three pinned
prompts: each initial score vector, captured embedding and 12 layer outputs,
and four teacher-forced score vectors per prompt. Each prompt uses one GPU-
policy request whose recurrent state advances through the four generated
tokens. Teacher tokens come from the portable full-prefix reference at each
step. The frozen score maximum-absolute-error limit is 0.04; ranking passes
when top-1 matches or the portable reference top-two margin is at most 0.08.
Hidden values pass when relative RMSE is at most 0.01 and cosine similarity is
at least 0.995. The runner also performs the existing 17 request and generation
lifecycle checks. It reports all results, including a completed numerical
no-go, without changing these thresholds.

This is only the short gate. It does not run full prompt generation, the
partition, boundary, completed-generation, or performance suites. Those
remain contingent on a passing short numerical/lifecycle result and a separate
decision. A numerical no-go is a completed experiment and must not be described
as a process failure. An identity, guard, execution, or preservation error
marks the attempt failed. Both statuses retain the latest atomic progress
record and error details.

The process records per-model construction time, comparison metrics, the
source/cache/runtime and adapter provenance before and after, prediction guard
counter deltas, and lifecycle outcomes. A final preservation check is required
before the attempt is marked complete. It makes no performance claim and no
global GPU-qualification claim.

Pinned source candidate aggregate:
`5a3e4c7a571e4fc3532377d9d9bb0da023f499c10709216fee7e1001bf4ad2c6`.

Pinned compiled aggregate:
`a838dd36e8f87c6b9fc0dd7865d0912651c6ab3399fa299d10ff12d788bcb031`.

Pinned portable acceptance SHA-256:
`cb4d315266126df4a98194f73f8c10671310717c06ebfd191d3dac2334bf4233`.

Pinned short-suite input fixture SHA-256:
`54ce299bdc107b3db80a2a389283ea787f7801a4f51a1ae2feed0a6dfaee7490`.
