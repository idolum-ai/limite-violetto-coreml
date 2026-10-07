# Evidence index

These are **recorded September 22–24, 2026 experiments**, imported on October 7. They are not fresh inference results. Paths in selected report copies are redacted; raw captures, logs, binary artifacts, and host-bound compiled identities remain in the local archive. Read [archive and integrity boundaries](../docs/archive.md) before checking historical hashes or attempting replay.

For the current conclusion, start with the [CPU promotion decision](05g-integration/decision.json), the [practical CPU comparison](05-practical/REPORT.md), the [failed full GPU screen](06b-full-gpu/REPORT.md), and the [latest GPU localization](06g-gpu-residual/REPORT.md). Original NVIDIA parity, ANE, GPU correctness, actual GPU placement, and clean-machine installation remain unqualified.

## Investigation history

Each row describes that phase's result. Later phases supersede earlier local failures only where their reports explicitly establish it; historical failures remain useful evidence.

| Phase | Report | Recorded outcome / useful record |
|---|---|---|
| 01 · Reference acquisition | [Sources and reference](01-reference/REPORT.md) | Pinned sources; original NVIDIA oracle unavailable. [Baseline specification](01-reference/BASELINE-CAPTURE.md) |
| 02 · Component conversion | [Components](02-components/REPORT.md) | Six graphs pass; corrected RMS epsilon. [Frozen criteria](02-components/ACCEPTANCE.json) |
| 03 · Portable model | [Initial portable model](03-portable/REPORT.md) | Full-prefix passes; initial cache gate fails. [Frozen criteria](03-portable/ACCEPTANCE.json) |
| 03b · Portable CPU cache | [Cache follow-up](03b-cache/REPORT.md) | CPU cache/partition/generation gates pass; Metal and NVIDIA parity open |
| 04 · Full Core ML | [First complete conversion](04-coreml/REPORT.md) | Complete generation executes, numerical fidelity fails |
| 04b · Cleanup reliability | [Cleanup crash](04b-reliability/REPORT.md) | Specific cause diagnosed and bounded guard regressions pass; raw crash trace excluded |
| 04c · Ordered precision | [Precision follow-up](04c-precision/REPORT.md) | Local ordered linear parity; full fidelity still open at this stage |
| 04d · CPU qualification | [CPU report](04d-cpu/REPORT.md), [audit](04d-audit/REPORT.md) | Candidate 6 passes 54 short, 97 partition, six boundary comparisons and 674 generated tokens. [Milestone](04d-cpu/candidate6-milestone.json) |
| 04e · Prediction deadlock | [Runtime guard](04e-runtime/REPORT.md) | Process-local guard qualified on the pinned host |
| 04f · Tiny prefixes | [Tiny-prefix arithmetic](04f-tiny/REPORT.md) | Operator, full numerical and completed-generation gates pass |
| 05 · Practical evaluation | [Pilot](05-practical/REPORT.md) | Earlier portable CPU 12–15 tokens/s vs Core ML about 0.4; portable recommended |
| 05b · Compiled startup | [One-group cache](05b-startup/REPORT.md) | Startup/replay parity passes; expanded in 05c |
| 05c · Persistent full cache | [Full compiled cache](05c-compiled/REPORT.md) | Full pinned-CPU qualification passes; 13.9–14.3s cache-conditioned reopen |
| 05d · Decode profiling | [Profile](05d-decode/REPORT.md) | 84 exact score comparisons; backbone calls account for 97.97% of profiled time |
| 05e · Value-product ablation | [Product experiment](05e-product/REPORT.md) | 180 exact calls; performance no-go, only 1/9 cells meets gate |
| 05f · Warm-score ablation | [Score experiment](05f-scores/REPORT.md) | 180 exact calls and 9/9 timing cells pass; integration follows in 05g |
| 05g · CPU integration | [Promoted CPU candidate](05g-integration/REPORT.md) | Full gates and 674-token parity pass; [promotion decision](05g-integration/decision.json) |
| 06 · GPU policy probe | [Original group0](06-gpu/REPORT.md) | Strict parity fails despite five outputs within the diagnostic relative-error bound |
| 06b · Full GPU screen | [Full-model screen](06b-full-gpu/REPORT.md) | Numerical no-go: 15/15 score checks fail; 17/17 lifecycle checks pass |
| 06c · Drift localization | [Raw replay](06c-gpu-localization/REPORT.md) | Controlled captures isolate halfway rounding; endpoint/cache controls pass |
| 06d · Rounding correction | [Separate correction](06d-gpu-rounding/REPORT.md) | First Q/K/V exact across policies; group0 error reduced, residuals remain |
| 06e · Normalization diagnosis | [Equal-input probe](06e-gpu-normalization/REPORT.md) | Three one-ULP reciprocal differences explain 19 rounded outputs; first proposed correction rejected |
| 06f · Bounded reciprocal | [One-site correction](06f-gpu-reciprocal/REPORT.md) | 19 warm differences removed; CPU preserved; standalone seed reproduction and residuals remain open |
| 06g · Remaining key cache | [Latest localization](06g-gpu-residual/REPORT.md) | 24 native calls / 12 control cells pass; one key error localized; [scaled native correction planned, not run](06g-gpu-residual/next-experiment.json) |

## Reading and replaying records

`REPORT.md` provides interpretation; `commands.md` and protocol/design files describe the corresponding execution. JSON records include criteria, identities, gate outcomes, and failures. A process exiting with code zero does not establish a numerical pass. Some old commands use fixed output names and must be inspected before reuse.

Historical relative links into omitted logs, captures, crash traces, source snapshots, or compiled identities refer to the local archive. Likewise, historical checksum entries describe original bytes and may refer to omitted or redacted material. The five `SHA256SUMS` files from 06c through 06g omit private trace entries and are partial public exports, not complete original seals. [import-manifest.json](../docs/import-manifest.json) and `scripts/verify_repository.py` define the export's integrity boundary.

Fresh cleanup checks are documented separately in [validation.md](../docs/validation.md); they do not upgrade any research qualification status.
