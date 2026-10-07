# Full persistent compiled cache qualification

Status: COMPLETE PASS for the pinned local CPU environment. The explicit compiled loader preserves candidate 6 fidelity and substantially reduces observed startup after the first opening. Inference, arithmetic, request behavior, native guards and numerical limits are unchanged. The default loader is unchanged.

## Validation gates

| Gate | Result | Evidence |
|---|---|---|
| C1: build and identity | All 14 components compile; complete compiled bytes, source packages and recorded environment/toolchain verify | [Build](full-cache-1/status.json), [identity](full-cache-1/compiled-identity.json) |
| C2: fresh-process startup | All 4 probes match first 32-token prefill and following decode scores exactly; 28 guarded calls and clean disposal per probe; normal exits | [Validated summary](startup-summary.json), [fixture binding](startup-fixture-binding.json) |
| C3: short suite | 54/54 numerical and 17/17 lifecycle checks pass | [Results](short/validation-CPU_ONLY.json), [summary](short/summary.json) |
| C4: partitions and capacity | 97/97 partition checkpoints across 13 cases; 13 disposals and exact alternate-partition repeat; 6/6 boundary cases and 4 lifecycle/repeat checks pass | [Partitions](../04d-audit/compiled1-partitions/summary.json), [boundaries](boundaries/summary.json) |
| C5: completed generation and combined provenance | All 674 tokens match both qualified CPU and package-loader runs; correct boxed 391, explicit stop 151643, state disposed, 7/7 score audits pass; all identities agree | [Generation](../04d-audit/compiled1-generation/results.json), [token parity](generation-loader-parity.json), [combined qualification](milestone-rerun1.json) |

All 142 hidden comparisons across the short, partition and boundary suites are exact. Maximum observed score error is 3.814697265625e-6 under the unchanged 0.04 limit. All inference processes exited normally. [Recorded exits](process-exits.json).

## Startup observations

Sequential fresh processes; OS/filesystem caches were not controlled. The first opening of the new compiled path is reported separately from its later reopens. The two-reopen median readiness is 14.117s versus 240.406s for the one package control: approximately 17.0x faster, or 94.1% less time. Median time to first scores is 25.729s versus 247.521s, approximately 9.6x faster for this 32-token probe.

| Process | Entry to ready (s) | Entry to first scores (s) | Peak process RSS (GiB) |
|---|---:|---:|---:|
| Package control | 240.406 | 247.521 | 6.721 |
| First compiled opening | 220.675 | 227.816 | 5.778 |
| Compiled reopen 1 | 13.907 | 28.103 | 5.090 |
| Compiled reopen 2 | 14.328 | 23.354 | 5.627 |

Explicit compiler calls total 21.475s; the side-by-side cache contains 5,049,395,550 bytes (4.703GiB). Compiled reopening includes full source/cache identity checks; their roughly 7-second cost is included in readiness. The two loaders intentionally perform different identity work, matching their actual invocation paths.

These are descriptive observations with one package control and two later reopens, not cold-machine or cache-eviction measurements. RSS is the process high-water mark, not steady-state or system-wide memory. There is no established decode speedup. Long qualification processes overlapped and their elapsed times must not be used as performance benchmarks.

## Delivery and preservation

The new explicit CLI, `python -m coreml_port.compiled_generate`, requires both source and compiled identities. Its two-token smoke test verified the exact prompt and sampled reference tokens, streaming, sampling seed, explicit EOS and intentional length limit. [Assertions](cli-smoke-checks.json), [output](cli-smoke.json), [commands](commands.md).

All 20 original arithmetic/runtime/guard identities remain unchanged. All 10 new loader/validator cohort files and their snapshots remain frozen. Previous validator versions are preserved; edits only add selection and provenance. [Original preservation](qualified-source-preservation.json), [cohort verification](cohort-source-verification.json), [previous validators](previous-validator-sources/identity.json), [independent review](independent-review.md).

Synthetic checks: 12 identity rejection/roundtrip checks and 14 aggregation tests pass. [Identity tests](identity-tests.log), [aggregation tests](aggregator-rerun-tests.log). The first end-to-end aggregate found a reporting-code argument error: source-code hashes were supplied where the package manifest was required. The failed report and source are preserved; correcting that one argument made the same frozen evidence pass full verification. No model, runtime, validator, evidence or threshold was changed to resolve it. [Failed first aggregate](milestone.json), [original aggregator](aggregator-first-attempt.py), [passing rerun](milestone-rerun1.json).

This qualifies the recorded scope on the current macOS 26.5.2 arm64 M4 / Python 3.12.9 / Core ML Tools 9.0 environment with its pinned native guards. It does not qualify GPU, Neural Engine, original NVIDIA-runtime fidelity, clean-machine installation or broad task accuracy.

## Next milestone

Profile warm decode with fixed streams and component timings, preserving this qualified baseline. Identify the dominant computation before choosing one isolated optimization. Repeat parity checks and a controlled throughput comparison for any proposed change. Portable CPU remains the practical inference recommendation; accelerated-device work remains separate.

Handoff: [checksums](SHA256SUMS), [verification](checksum-verification.json). Historical checksums and failed experiments remain preserved.
