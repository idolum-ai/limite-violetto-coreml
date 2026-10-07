# Persistent compiled loading — successful isolated experiment

**Persistent compiled reopening is worth extending to the full pipeline.** Sealed candidate6 group0 takes 18.7–18.8 seconds to construct through `MLModel(.mlpackage)`. The first `CompiledMLModel(.mlmodelc)` opening takes 16.64 seconds, while subsequent openings in fresh processes take 0.35–0.36 seconds. The median constructor ratio is approximately 53× for subsequent openings. This is a measured single-group result, not a full-pipeline startup claim.

Explicit compilation took **1.863 seconds**, producing **278,311,875 bytes (265.42 MiB)** at `artifacts/05b-startup/group0-1/0.mlmodelc`. Compilation and first device-specific opening are separate costs; the short explicit compile did not remove the first opening's 16.64-second preparation cost.

## Fresh-process observations

| Order | Loader | Constructor | State creation | Entry to loaded |
|---|---|---:|---:|---:|
| 1 | package | 18.668 s | 0.0418 s | 20.345 s |
| 2 | compiled | 16.639 s | 0.0025 s | 18.286 s |
| 3 | compiled | 0.351 s | 0.0024 s | 2.008 s |
| 4 | package | 18.838 s | 0.0377 s | 20.478 s |
| 5 | package | 18.764 s | 0.0361 s | 20.406 s |
| 6 | compiled | 0.362 s | 0.0025 s | 2.005 s |

Every row is a separate child process. Both loaders hash the same source and compiled artifacts, inspect the same source specification, and install both pinned guards before the constructor timer. Entry-to-loaded also includes host observations, Python/library/guard setup and identity checks; it does not include state creation. Constructor time excludes that setup. No filesystem, framework or device caches were evicted. The persistent artifact path is reused, so these are cache-conditioned observations, not cold-start measurements or a guaranteed cache lifetime.

The expensive first compiled opening followed by fast reopenings is consistent with reuse of preparation work tied to the persistent artifact. The experiment does not identify Core ML's private cache mechanism or guarantee behavior after eviction, reboot or an OS/toolchain change. Three observations per loader are descriptive, with the first compiled opening reported separately.

## Prediction and state parity

Two further fresh-process workers load the package and compiled forms, respectively. Each restores two independent states from the saved native group0 prefix and predicts the 32-query capture at positions 1058–1089.

- Both loaders match native y bitwise, with zero maximum error.
- Independent restored-state repeats have exact output and post-call cache equality for each loader.
- Package and compiled outputs match exactly, as do all eight post-call KV state tensors.
- Both workers observe two prediction-guard state calls and two GIL releases; both exit normally after releasing Python model/state references. This is bounded replay evidence, not proof of every native lifetime behavior.
- Package prediction calls take 0.557/0.510 seconds; compiled calls take 0.563/0.549 seconds. These two captured 32-query calls show no meaningful prediction advantage and are not a decode benchmark. Loading reuse does not establish a fix for the earlier measured decode gap.
- Original package files, compiled artifact files and qualified runtime identities remain unchanged throughout.

[Package replay](group0-1/replay-package/results.json), [compiled replay](group0-1/replay-compiled/results.json), [cross-loader comparison](group0-1/results.json), [independent review](independent-review.md).

## Validation gates and reproducibility

| Gate | Status | Evidence |
|---|---|---|
| C1 Explicit compilation and identity | PASS | [Compile record](group0-1/compile/results.json) |
| C2 Fresh-process construction and state API | PASS for bounded observations | [Attempt with all child commands](group0-1/results.json) |
| C3 Native and cross-loader prediction/cache parity | PASS for saved stateful capture | Replay links above |
| C4 Next-step decision | Extend as an isolated opt-in loader; require full-pipeline qualification | This report |

[Machine-readable summary](summary.json), [exact probe source snapshot](group0-1/source-snapshot.py), [reproduction commands](commands.md), [handoff checksums](SHA256SUMS).

## Next milestone

Create a side-by-side compiled cache for all 14 sealed packages, with a manifest binding compiled bytes to the source package identity, Core ML/macOS pins and required input/state metadata. Test an opt-in full-pipeline loader; preserve candidate6, its default selection and runtime qualification. `CompiledMLModel` lacks `get_spec()`, so source-spec inspection or explicit verified metadata must supply `norm_divisor` and state information.

Measure full-process readiness with and without the cache, then run the existing numerical and lifecycle gates under the new loader's recorded runtime identity before adoption. The one-group result cannot be multiplied into a promised 14-package startup time. After that, separately profile the expensive warm prediction path. GPU and Neural Engine execution remain unqualified.
