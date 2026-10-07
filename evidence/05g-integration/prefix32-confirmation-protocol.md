# One P32 confirmation

This is one bounded follow-up to the original full benchmark's P32 round
variation and observed AC-to-battery change. The original source, protocol,
results and passing full-1 milestone remain unchanged and are pinned by the
new JSON protocol. No native job was run during development.

```sh
.venv/bin/python scripts/confirm_conditional_prefix32.py \
  --attempt prefix32-confirmation-1 \
  --protocol evidence/05g-integration/prefix32-confirmation-protocol.json \
  --candidate-milestone evidence/05g-integration/full-1/milestone.json \
  --confirm-isolated
```

Run with no other native model, GPU or benchmark work. Four fresh subprocesses
run A-B-B-A, one complete pipeline resident at a time. Each worker performs
an excluded 32-token prefill plus three forced decode steps, then a fresh
measured 32-token prefill plus three forced decode steps. The qualified fixed
prompt/reference stream, unchanged loader and guards, and explicit source/cache
cohort are inherited from the pinned original benchmark helper/protocol.

Only synchronous full pipeline forward is timed. Loading, verification, state
creation/disposal, score copying/saving/comparison and host telemetry are
excluded. Expected worker guard counts are 16 plain, 96 stateful and 112 GIL
releases. Both requests must be disposed. Parent and workers retain source and
protocol snapshots, helper snapshots, identities, telemetry, timings, score
archives and exit records.

All four measured vectors per worker must match both the first A worker and
the independently pinned historical P32 pilot fixture, including dtype, shape
and literal array bytes. This distinguishes signed zero and dtype differences.

The eight worker before/after-timing power-source first lines must all report
the same recognized source: `Now drawing from 'AC Power'` or
`Now drawing from 'Battery Power'`. No source is forced. Missing/unrecognized
telemetry, an observed transition, or nonstandard/recorded thermal/performance
warning status stops the attempt. Both standard negative messages are required:
`Note: No thermal warning level has been recorded` and
`Note: No performance warning level has been recorded`. The standard negative
CPU power-status message is also allowed. Full telemetry is retained.

Performance passes only at least 10% median decode reduction and no more than
10% median prefill regression. There are two prefill and six decode samples per
arm. This remains a local, small-sample measurement; it does not remove OS-cache
or scheduling uncertainty.

This protocol allows only the fixed fresh attempt `prefix32-confirmation-1`.
A numerical, host-stability, process or timing failure ends confirmation this
turn. The candidate remains correctness-qualified but is not promoted. There
is no further retry under this plan.

Development checks passed for protocol/cohort pins, inclusive 10% boundaries,
9% improvement rejection, stable AC and stable battery observations, rejection
of transitions/missing/warning status, and signed-zero/dtype exactness. These
checks performed no native model prediction.
