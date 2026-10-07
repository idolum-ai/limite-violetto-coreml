# Whole-pipeline conditional benchmark

Run only after the new CPU candidate passes its complete correctness milestone
and other native model/GPU/benchmark jobs have finished. The executable protocol
is `benchmark-protocol.json`; it pins the benchmark and qualification-helper
source hashes, qualified teacher-forced fixture, explicit source/cache paths,
ABBA schedule and performance criteria.

```sh
.venv/bin/python scripts/benchmark_conditional_pipeline.py \
  --protocol evidence/05g-integration/benchmark-protocol.json \
  --candidate-milestone evidence/05g-integration/full-1/milestone.json \
  --attempt pipeline-abba-1 \
  --confirm-isolated
```

The selected milestone filename can differ; it must be the passing conditional
aggregator result bound to the selected full-1 source/cache. Existing evidence
is never overwritten.

Eight fresh subprocesses execute A-B-B-A twice. Only one complete pipeline is
resident at a time. A is the qualified baseline compiled loader; B is the new
single-model conditional candidate using that unchanged loader and both native
guards. Every worker verifies its source/cache bytes before loading.

Each worker first performs an excluded 32-token prefill and three forced
decode steps. It then uses a fresh request for each measured prefix (32, 128,
512), followed by three forced decode steps. Round two reverses prefix order.
The stream is the frozen qualified prompt followed by all 674 reference token
IDs. It never depends on candidate predictions or sampling.

Only synchronous full `Pipeline.forward` is timed. Normal 32-token prefill
chunking is included. Manifest verification, model construction, state
creation/disposal, score copying/saving/comparison and host telemetry are
outside those timers. The result separately reports initialization, component
loading, verification and request costs. All 12 measured score sites in each
worker are compared for exact array values with the first A worker, including repeated A
runs; every request must be disposed. Expected guard counts per worker are
50 stateless predictions, 408 stateful predictions and 458 GIL releases.

A completed performance go requires all correctness/provenance checks plus:

- At least 10% median decode improvement at each prefix and over all decode
  samples pooled.
- At most 10% median prefill regression at every prefix.

Each prefix has four prefill observations and twelve decode observations per
arm. Per-round decode changes, load average, power and thermal state are
reported. A difference over ten percentage points between rounds is flagged.
These are local measurements from two ABBA rounds, with uncontrolled OS caches
and host scheduling; inspect the noise evidence before promoting a candidate.
A completed negative timing result is a valid no-go, not an execution failure.

Cost estimate from retained baseline measurements: prefill medians were
6.67/26.89/109.15 seconds for 32/128/512 tokens, with decode about 2.31 seconds
and compiled startup about 14 seconds. If the new candidate has no speedup,
this plan takes approximately 25–30 minutes; the eight 512-token prefills
account for about 14.6 minutes. This estimate is historical, not a new run.

Development checks run no native prediction. The summary accepts a synthetic
20% decode improvement with 5% prefill regression, and rejects a single-prefix
9% decode improvement, 11% prefill regression, or missing timed sample.
