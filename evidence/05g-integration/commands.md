# CPU integration reproduction

Run from the Violetto directory with the pinned `.venv`. Native Core ML operations require ordinary host execution. Use fresh evidence/attempt paths; preserve prior results.

Final group builds use `scripts/build_conditional_scores.py`, group0 attempt `conditional-group0-3` and groups1–11 attempt `conditional-groupN-2`; `cohort2-group-records.json` lists the exact records. The full candidate was assembled with:

```sh
.venv/bin/python scripts/assemble_conditional_model.py --attempt full-1 --records-json evidence/05g-integration/cohort2-group-records.json
```

All full validators use this exact selection:

```text
--model-dir artifacts/05g-integration/full-1/source
--identity-manifest evidence/05g-integration/full-1/package-identity.json
--compiled-dir artifacts/05g-integration/full-1/compiled
--compiled-manifest evidence/05g-integration/full-1/compiled-identity.json
--tiny-prefix-replay
```

Append the selection to these existing unchanged scripts and supply fresh output directories:

| Script | Recorded output target |
|---|---|
| `scripts/validate_coreml.py` | `--evidence-dir evidence/05g-integration/short` |
| `scripts/audit_coreml_partitions.py` | `--evidence-dir evidence/04d-audit/conditional1-partitions` |
| `scripts/validate_coreml_boundaries.py` | `--evidence-dir evidence/05g-integration/boundaries` |
| `scripts/audit_coreml_generation.py` | `--evidence-dir evidence/04d-audit/conditional1-generation` |

The generation and partition scripts require output below `evidence/04d-audit/`. Qualification jobs may overlap; such timings are not performance measurements. The [whole-pipeline benchmark](benchmark-protocol.md) must run by itself, only after the correctness milestone passes. This document records intended/issued commands; actual completion is determined by the [gate report](REPORT.md) and linked result files.


The complete qualification aggregate is `full-1/milestone.json`. The [full benchmark command](benchmark-protocol.md) and [one-shot confirmation command](prefix32-confirmation-protocol.md) each bind their fixed protocol. The confirmation is deliberately not rerunnable under a new attempt label; a future study needs its own declared protocol.

Final preservation verification was run after all native experiments:

```sh
.venv/bin/python scripts/verify_conditional_preservation.py --output evidence/05g-integration/final-preservation.json
```

Use a fresh output path if reproducing that read-only verification. Promotion selects the new sealed source/cache through the unchanged compiled CLI; the recommended command and original compiled fallback are in the root README.
