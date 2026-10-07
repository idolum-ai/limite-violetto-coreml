# Archive, privacy, and integrity boundaries

The repository is a curated export of a local research archive. The source archive's files were read but not edited, moved, or deleted during cleanup. The archive location is deliberately omitted from tracked records; the owner retains it locally.

## What Git contains

Research source, pinned licensed upstream snapshots, tokenizer/configuration assets, frozen acceptance criteria, reports, numerical JSON, and historical checksum lists. Failed experiments remain visible alongside passes. [import-manifest.json](import-manifest.json) records each imported path, its original SHA-256, its exported SHA-256, and its treatment.

117 imported files had local home/project/temp paths replaced by `<USER_HOME>`, `<PROJECT_ROOT>`, or `<TEMP_ROOT>`. The approved public preparation additionally replaced 58 host-local battery ID values in 29 JSON files with `<REDACTED_HOST_ID>` and omitted 660 private trace entries across five historical checksum lists. These sets overlap. Power-source, battery percentage, timing, thermal, model-result and acceptance data are unchanged. These are report exports, not new experiment results. One synthetic test was also made independent of a local native binary; that edit is separately marked in the import manifest.

**Historical embedded hashes and remaining `SHA256SUMS` entries describe original archive bytes.** They may refer to omitted files or pre-redaction content. The checksum files under `06c-gpu-localization`, `06d-gpu-rounding`, `06e-gpu-normalization`, `06f-gpu-reciprocal`, and `06g-gpu-residual` are partial public exports: private `.trace/` entries are omitted. They are not complete original seals. Do not run those lists against this export and treat a failure as corruption of the underlying experiment. `scripts/verify_repository.py` verifies the export's identities; verifying the full historical seals requires the untouched archive.

Host-bound `compiled-identity.json` files and a crash-stack record are omitted outright. Compiled manifests are never redacted and reissued as valid runtime identities. Source package manifests and research summaries remain available for inspection, but model packages and executable identities must be restored from the archive before running a candidate.

## What remains local

| Material | Reason / restoration |
|---|---|
| `artifacts/` (about 78.6 GB in the source inventory) | Generated Core ML source/compiled packages and native binaries; restore only the cohort an experiment needs |
| `sources/model/model.safetensors` (2.07 GB) | Large upstream weights; use the pinned retrieval and hash instructions in [setup.md](setup.md) |
| Tensor captures (`.npz`, `.npy`) | Roughly 907 MB of evidence captures; needed for exact saved-tensor replay and counterfactuals |
| Native logs, `.trace` bundles, crash dumps, operator trail, sessions | Machine/private operational material; not publication inputs |
| Duplicate source snapshots and acquisition payloads | Preserved locally; the export contains the current source and relevant report records |
| `sources/autoresearch`, `sources/xi-log-concavity-lean`, `sources/coremltools-main-bridge` | Ancillary copies lack a complete retained license/provenance package; excluded pending review |
| `.venv`, caches, credentials | Machine-local state; recreate dependencies from the hash lock |

The exclusion totals in the import manifest are inventory counts/sizes, not a fresh hash verification of tens of gigabytes. No large model was converted, downloaded, or rerun during cleanup.

The first local repository candidate is preserved in a private Git bundle under the ignored maintenance directory. Public history starts from the sanitized tree and does not include that earlier candidate as an ancestor. Private bundles and the original raw archive are not publication inputs. Personal attribution was retained with the owner's approval; upstream library build identifiers remain where they support technical provenance.

## Restoring for an experiment

Use a separate working copy and restore selected files at their original relative paths from the owner's trusted local archive. Before copying, inspect the relevant historical identity manifest and checksum list. Keep the original unredacted evidence together with its artifacts so the old provenance relationships remain intact.

For the promoted CPU path, the relevant groups are:

```text
artifacts/04b-reliability/cleanup_guard.dylib
artifacts/04e-runtime/prediction_guard.dylib
artifacts/05g-integration/full-1/source/
artifacts/05g-integration/full-1/compiled/
evidence/05g-integration/full-1/package-identity.json
evidence/05g-integration/full-1/compiled-identity.json  (raw archive only)
```

Copying into a different absolute environment is not sufficient to qualify a compiled cache. The original archive is the simplest place to inspect the recorded candidate. For a fresh checkout, rebuild/rebind and revalidate rather than weakening identity checks. New runs should write new evidence and preserve historical results.

The legacy `verify_sources.py` checks the full acquisition archive and overwrites its integrity record; `verify_arithmetic_sources.py` additionally requires local bridge/guard binaries. Neither is a clone-only verification command. Their original source is retained to make the earlier investigation understandable; use the new repository verifier for the exported subset.
