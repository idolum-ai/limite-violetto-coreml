# Precision milestone reproduction

Run from `<PROJECT_ROOT>`. Core ML prediction needs host access on this machine. Each current candidate runner installs the version/hash-pinned cleanup guard followed by the version/hash-pinned prediction guard before loading models. No tolerance was changed.

The isolated attention harness refuses an existing run directory. Its captured inputs come from the first canonical prompt's 32-token prefix, layer 1, with ordered linear arithmetic. The first attempt had an evidence filename collision; use the corrected later evidence.

```sh
.venv/bin/python scripts/probe_cpu_attention.py --run NEW_ATTEMPT
```

The uninstrumented corrected shard and Torch full graph were built/tested with:

```sh
.venv/bin/python scripts/convert_coreml.py 0 --arithmetic ordered --softmax ordered --output-dir artifacts/04d-cpu/ordered-softmax --evidence-dir evidence/04d-cpu/ordered-softmax-shard
.venv/bin/python scripts/probe_coreml_shard.py --arithmetic ordered --softmax ordered --compute CPU_ONLY --model-dir artifacts/04d-cpu/ordered-softmax --evidence-dir evidence/04d-cpu/ordered-softmax-shard
.venv/bin/python scripts/ablate_graph_precision.py --mode ordered --softmax ordered --evidence-dir evidence/04d-cpu/ordered-softmax-shard
.venv/bin/python scripts/probe_coreml_layer_stages.py --softmax ordered --compute CPU_ONLY --evidence-dir evidence/04d-cpu/ordered-softmax-stages --artifact-dir artifacts/04d-cpu/ordered-softmax-stages
```

Those scripts may overwrite/reuse fixed outputs. Preserve the existing attempt and select new paths before rerunning changed arithmetic. Diagnostic outputs can change compiler fusion; the stage probe is not a gate pass.

The complete trial uses two isolated converter subprocesses at a time. Its build script refuses existing output directories, copies the unchanged embedding and already-converted shard 0, then converts groups 1–11 and the ordered head. It records each command/exit code and inventories the complete candidate.

```sh
.venv/bin/python scripts/build_cpu_candidate.py
.venv/bin/python scripts/seal_cpu_candidate.py
.venv/bin/python scripts/validate_coreml.py --compute CPU_ONLY --model-dir artifacts/04d-cpu/full-candidate-1 --identity-manifest evidence/04d-cpu/full-candidate-1/package-identity.json --evidence-dir evidence/04d-cpu/full-candidate-1
```

The sealing step checks conversion-source stability and digest equality of copied packages, and records runtime/validator identities. Explicit alternate package selection requires both root and manifest. The full validator still has the original frozen numerical criteria; longer cache/window checks follow only if this trial justifies them.

Subsequent attempts select corrections explicitly and retain separate roots:

```sh
.venv/bin/python scripts/build_cpu_candidate.py --attempt full-candidate-2 --enhanced
.venv/bin/python scripts/seal_cpu_candidate.py --attempt full-candidate-2
.venv/bin/python scripts/validate_coreml.py --compute CPU_ONLY --model-dir artifacts/04d-cpu/full-candidate-2 --identity-manifest evidence/04d-cpu/full-candidate-2/package-identity.json --evidence-dir evidence/04d-cpu/full-candidate-2
.venv/bin/python scripts/probe_sleef_exp.py --run sleef-exp-2 --modes double,dd
.venv/bin/python scripts/validate_source_exp.py --attempt source-exp-independent-1
.venv/bin/python scripts/build_cpu_candidate.py --attempt full-candidate-3 --enhanced --source-exp
```

Use new attempt names to rerun. Candidate 3 copies embedding/head from candidate 2; all group conversions select `--exp source`. The exponential probe is bounded to nonpositive softmax inputs, including masked negative infinity. Historical candidate source snapshots preserve the code used for their conversions.

Candidate 3 completed conversion and short-gate validation successfully:

```sh
.venv/bin/python scripts/seal_cpu_candidate.py --attempt full-candidate-3
.venv/bin/python scripts/validate_coreml.py --compute CPU_ONLY --model-dir artifacts/04d-cpu/full-candidate-3 --identity-manifest evidence/04d-cpu/full-candidate-3/package-identity.json --evidence-dir evidence/04d-cpu/full-candidate-3
.venv/bin/python scripts/validate_coreml_boundaries.py --model-dir artifacts/04d-cpu/full-candidate-3 --identity-manifest evidence/04d-cpu/full-candidate-3/package-identity.json --evidence-dir evidence/04d-cpu/full-candidate-3-boundaries
```

The CLI accepts `--model-dir` together with `--identity-manifest`, verifies the complete package set before loading, and includes the selected root and aggregate digest in its output. The historical default is still unchanged. CLI help and incomplete-argument rejection are retained as `candidate-cli-help.txt` and `candidate-cli-argument-check.json`.

## Query geometry correction and candidate 4

Run from the project root. Core ML prediction requires host framework access; sandbox-only load failures are not numerical evidence. Every candidate validation below selects the same opt-in CPU tiny-prefix replay adapter. The following commands record the **historical candidate-4 build** exactly; its output paths are already occupied and the builder refuses to overwrite them. The earlier candidate4 boundary cohort passes 1023/1024/1025/1026 and fresh1026 but fails capacity2048. The old-guard partition run deadlocked and is preserved only as diagnostic evidence.

```sh
.venv/bin/python scripts/build_cpu_candidate.py --attempt full-candidate-4 --enhanced --source-exp --dynamic-product
.venv/bin/python scripts/seal_cpu_candidate.py --attempt full-candidate-4
.venv/bin/python scripts/validate_coreml.py --compute CPU_ONLY --model-dir artifacts/04d-cpu/full-candidate-4 --identity-manifest evidence/04d-cpu/full-candidate-4/package-identity.json --tiny-prefix-replay --evidence-dir evidence/04d-cpu/full-candidate-4
.venv/bin/python scripts/audit_coreml_partitions.py --model-dir artifacts/04d-cpu/full-candidate-4 --identity-manifest evidence/04d-cpu/full-candidate-4/package-identity.json --tiny-prefix-replay --evidence-dir evidence/04d-audit/candidate4-partitions
.venv/bin/python scripts/validate_coreml_boundaries.py --model-dir artifacts/04d-cpu/full-candidate-4 --identity-manifest evidence/04d-cpu/full-candidate-4/package-identity.json --tiny-prefix-replay --evidence-dir evidence/04d-cpu/full-candidate-4-boundaries
```

### Candidate-4 runtime2 status

On the qualified macOS 26.5.2 arm64 / CPython 3.12.9 host with the pinned Core ML Tools 9.0 bridge, both native guards must be built and installed before models load. Build the cleanup guard first and prediction GIL guard second. The runtime2 short rerun passes 54/54 comparisons and 17/17 lifecycle checks. Its full partition rerun completes without the old deadlock but fails 41/97 matched prefixes (56 pass); direct prefix3 and prefixes3–14 in cached partition fixtures fail. Frozen thresholds are unchanged. See the [actual-shard guard qualification](../04e-runtime/REPORT.md), [runtime2 short result](candidate4-runtime2-short/summary.json), [runtime2 partition result](../04d-audit/candidate4-runtime2-partitions/results.json), and [incomplete runtime2 milestone](candidate4-runtime2-milestone.json). The older candidate4 boundary result is from the earlier cleanup-only runtime cohort, so it cannot be combined with runtime2 to claim a complete boundary gate.

For a fresh run, after confirming the required host/pins and choosing unique evidence paths, build guards in this order:

```sh
.venv/bin/python scripts/build_cleanup_guard.py
.venv/bin/python scripts/build_prediction_guard.py
```

For a fresh rerun on macOS arm64, first create the CPython 3.12.9 environment with the [hash-locked installation command](../../README.md). The local cleanup guard requires Xcode command-line tools (`xcrun clang`) and the pinned Core ML Tools 9.0 bridge binary; its installer checks the bridge SHA256 before prediction. Build the guard only if its existing artifact is absent, then verify the retained arithmetic sources. The verifier is read-only unless given a new `--output` path.

```sh
xcrun --find clang
if [ ! -e artifacts/04b-reliability/cleanup_guard.dylib ]; then .venv/bin/python scripts/build_cleanup_guard.py; fi
.venv/bin/python scripts/verify_arithmetic_sources.py --output evidence/04d-gpu/arithmetic-source-verification-rerun1.json
.venv/bin/python scripts/build_cpu_candidate.py --attempt full-candidate-4-rerun1 --enhanced --source-exp --dynamic-product
.venv/bin/python scripts/seal_cpu_candidate.py --attempt full-candidate-4-rerun1
.venv/bin/python scripts/validate_coreml.py --compute CPU_ONLY --model-dir artifacts/04d-cpu/full-candidate-4-rerun1 --identity-manifest evidence/04d-cpu/full-candidate-4-rerun1/package-identity.json --tiny-prefix-replay --evidence-dir evidence/04d-cpu/full-candidate-4-rerun1
.venv/bin/python scripts/audit_coreml_partitions.py --model-dir artifacts/04d-cpu/full-candidate-4-rerun1 --identity-manifest evidence/04d-cpu/full-candidate-4-rerun1/package-identity.json --tiny-prefix-replay --evidence-dir evidence/04d-audit/candidate4-rerun1-partitions
.venv/bin/python scripts/validate_coreml_boundaries.py --model-dir artifacts/04d-cpu/full-candidate-4-rerun1 --identity-manifest evidence/04d-cpu/full-candidate-4-rerun1/package-identity.json --tiny-prefix-replay --evidence-dir evidence/04d-cpu/full-candidate-4-rerun1-boundaries
```

If any `rerun1` path already exists, choose a different suffix and substitute it consistently. The exact candidate-4 conversion sources are retained under `evidence/04d-cpu/full-candidate-4/source-snapshots`; the current `dynamic_queries.py` differs only by an explicitly recorded post-seal docstring correction.

The long failure can be replayed without recomputing its prefix:

```sh
.venv/bin/python scripts/replay_coreml_capture.py --package artifacts/04d-cpu/integrated-query-shard/8.mlpackage --inputs evidence/04d-cpu/long-group-locator/first-failure-inputs.npz --states evidence/04d-cpu/long-group-locator/first-failure-state-before.npz --evidence-dir evidence/04d-cpu/integrated-query-shard/long-replay
.venv/bin/python scripts/replay_coreml_capture.py --package artifacts/04d-cpu/integrated-query-shard/8.mlpackage --inputs evidence/04d-gpu/tiny3-productrows-group8/fixture.npz --evidence-dir evidence/04d-cpu/integrated-query-shard/tiny-replay
```

These commands document planned and completed steps; check their JSON results for actual completion and gate status. A zero process exit code alone is insufficient for the short validator.

## Candidate5 paired value products

Candidate5 selects paired value products only. The combined tiny-score policy regressed group8 and is held out; its failed preflight remains evidence. The builder snapshots the selected sources. Use a new attempt name on rerun. Complete acceptance still requires all four numerical/generation gates under the same runtime identities. Build and seal before running prediction.

```sh
.venv/bin/python scripts/build_cpu_candidate.py --attempt full-candidate-5 --enhanced --source-exp --dynamic-product --paired-product
.venv/bin/python scripts/seal_cpu_candidate.py --attempt full-candidate-5
.venv/bin/python scripts/validate_coreml.py --compute CPU_ONLY --model-dir artifacts/04d-cpu/full-candidate-5 --identity-manifest evidence/04d-cpu/full-candidate-5/package-identity.json --tiny-prefix-replay --evidence-dir evidence/04d-cpu/full-candidate-5
.venv/bin/python scripts/audit_coreml_partitions.py --model-dir artifacts/04d-cpu/full-candidate-5 --identity-manifest evidence/04d-cpu/full-candidate-5/package-identity.json --tiny-prefix-replay --evidence-dir evidence/04d-audit/candidate5-partitions
.venv/bin/python scripts/validate_coreml_boundaries.py --model-dir artifacts/04d-cpu/full-candidate-5 --identity-manifest evidence/04d-cpu/full-candidate-5/package-identity.json --tiny-prefix-replay --evidence-dir evidence/04d-cpu/full-candidate-5-boundaries
```

Run the generation audit only after short, partition and boundary numerical gates pass. Commands above are a reproduction recipe, not a claim that every gate passed; see each attempt's results.

## Candidate 6 native tiny-score arithmetic

Preflights pass; build in progress. Use a fresh attempt name on rerun and preserve logs. Seal and numerical gates must precede generation.

```sh
.venv/bin/python scripts/build_cpu_candidate.py --attempt full-candidate-6 --enhanced --source-exp --dynamic-product --paired-product --tiny-native > evidence/04d-cpu/full-candidate-6-build.log 2>&1
.venv/bin/python scripts/seal_cpu_candidate.py --attempt full-candidate-6
```

The matching aggregate policy is `paired-native-tiny` and runtime adapter is `tiny_prefix_replay`.

### Completed candidate 6 qualification

The following records the completed run. Output directories are already occupied; use new names for reruns. All prediction commands require the qualified host framework access and install both pinned guards. Numerical gates must finish and pass before generation.

```sh
.venv/bin/python scripts/validate_coreml.py --compute CPU_ONLY --model-dir artifacts/04d-cpu/full-candidate-6 --identity-manifest evidence/04d-cpu/full-candidate-6/package-identity.json --tiny-prefix-replay --evidence-dir evidence/04d-cpu/full-candidate-6 > evidence/04d-cpu/full-candidate-6-validation.log 2>&1
.venv/bin/python scripts/audit_coreml_partitions.py --model-dir artifacts/04d-cpu/full-candidate-6 --identity-manifest evidence/04d-cpu/full-candidate-6/package-identity.json --tiny-prefix-replay --evidence-dir evidence/04d-audit/candidate6-partitions > evidence/04d-audit/candidate6-partitions.log 2>&1
.venv/bin/python scripts/validate_coreml_boundaries.py --model-dir artifacts/04d-cpu/full-candidate-6 --identity-manifest evidence/04d-cpu/full-candidate-6/package-identity.json --tiny-prefix-replay --evidence-dir evidence/04d-cpu/full-candidate-6-boundaries > evidence/04d-cpu/full-candidate-6-boundaries.log 2>&1
.venv/bin/python scripts/audit_coreml_generation.py --model-dir artifacts/04d-cpu/full-candidate-6 --identity-manifest evidence/04d-cpu/full-candidate-6/package-identity.json --tiny-prefix-replay --evidence-dir evidence/04d-audit/candidate6-generation > evidence/04d-audit/candidate6-generation.log 2>&1
.venv/bin/python scripts/summarize_cpu_milestone.py --candidate-root artifacts/04d-cpu/full-candidate-6 --candidate-evidence-dir evidence/04d-cpu/full-candidate-6 --short-result evidence/04d-cpu/full-candidate-6/validation-CPU_ONLY.json --short-attempt evidence/04d-cpu/full-candidate-6/validation-CPU_ONLY-attempt.json --runtime-adapter tiny_prefix_replay --arithmetic-policy paired-native-tiny --partitions-result evidence/04d-audit/candidate6-partitions/results.json --boundaries-result evidence/04d-cpu/full-candidate-6-boundaries/results.json --generation-result evidence/04d-audit/candidate6-generation/results.json --output evidence/04d-cpu/candidate6-milestone.json
```

The aggregate returns `milestone_pass=true`, with no missing or failed gates. The earlier `candidate6-numerical-milestone.json` intentionally retains its pre-generation incomplete status.
