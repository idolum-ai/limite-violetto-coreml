# Cleanup validation · 2026-10-07

These checks validate the curated repository and bounded existing test harnesses. **No model inference, conversion, GPU run, large download, native guard build, or clean-machine installation was performed.** The historical CPU/GPU conclusions remain scoped to their recorded experiments.

| Check run during cleanup | Result |
|---|---|
| Imported-file integrity | 1,249 imported files match their declared export hashes; original hashes are retained alongside them |
| Pinned-source comparisons | 57 comparisons against retained Git/Hugging Face/source identity records pass |
| Syntax / structured evidence | 293 Python files parse; 767 JSON files parse |
| Local regression suite | `pytest scripts/test_*.py -q -p no:cacheprovider`: **39 passed** in 16.84 seconds |
| Compiled identity executable | **12 checks pass**, including tamper, extra-file, symlink, nonregular-file, missing-package, host-drift, and source-binding rejection |
| Upstream package contract | **12 passed** in 1.11 seconds; attention is stubbed, so this is not inference parity |
| Existing environment consistency | `pip check`: no broken requirements; CPython 3.12.9 on macOS 26.5.2 |
| Exact Git index audit | No excluded models/weights/captures/logs, private home paths, common credential patterns, unexpected binaries, or files above 16 MiB detected |
| New documentation links | All local navigation targets in newly written documentation resolve |

The tests used the already-installed local research environment, with bytecode/cache writes disabled for the test runs. Dependencies were neither installed nor upgraded. `pip check` reported a disabled pip cache because the sandbox does not permit its user-cache path; dependency consistency passed.

The largest included file is the upstream tokenizer at 11,422,267 bytes. The import preserves 283 original code/native/notice files with one documented change: the CPU milestone unit test now uses synthetic runtime hashes instead of reading an excluded native guard binary. All three retained `ACCEPTANCE*.json` files and model/runtime arithmetic sources are unchanged.

The exact-index scanner checks immutable staged blobs, not only working-tree files, and reports rule names and paths without printing secret matches. It covers common credential forms and private filesystem paths; it is not an exhaustive credential audit. Upstream attribution and license-contact information are retained. Ignore rules additionally protect weights, compiled packages, tensor captures, native outputs, raw logs, credentials, session material, and ancillary snapshots.

The initial full-import whitespace check found inherited formatting in historical evidence and licensed upstream snapshots. Those bytes are preserved; `.gitattributes` disables whitespace lint for those two archived trees and line-ending normalization globally. New and active local source/documentation remain subject to the staged whitespace check.

## Public preparation checks · same date, after owner approval

The initial publication pause was resolved by explicit owner approval of the reviewed metadata cleanup and public upload, with personal attribution retained. The original-contribution license statement remains unchanged; no new license grant is invented. The public repository is `idolum-ai/limite-violetto-coreml`; its identity and empty refs were checked again before public preparation.

The follow-up review found metadata outside the initial scanner's coverage: host-local battery IDs and private trace-session paths. The public export redacts 58 ID values in 29 JSON records and omits 660 trace entries from five partial checksum exports. Original archive bytes/hashes remain private and unchanged. Measurements and remaining checksum entries are preserved.

Fresh lightweight checks cover the updated import hashes, source comparisons, Python/JSON parsing, documentation links, exact staged blobs, and the public history's ancestry. `scripts/test_publication_audit.py` passes **four integration tests** using real temporary Git indexes, including a staged-private/working-tree-clean case and preservation of sanitized power data and a public library build UUID. The scanner now rejects these host/session metadata forms, trace artifacts and private Git bundles. These checks do not replace a comprehensive semantic secret audit.

The original local candidate is retained in an ignored private Git bundle. The public history starts at the sanitized root; merely appending redactions to the earlier commit would still disclose its data and is not the publication procedure. After committing, fetch again and push only the approved branch without force, then compare the remote SHA and inspect any triggered CI. No GitHub Actions workflow is introduced by this repository.

The earlier 39 local tests, 12 compiled-identity checks, and 12 upstream contract tests were not repeated for metadata-only public preparation. No numerical source changed. The syntax inventory now includes 294 Python files; the 767 JSON files and 1,249 import records remain intact with updated export identities.
