# Pinned source snapshots

These files are retained to make the local implementation and arithmetic decisions inspectable. They are not a replacement for the upstream projects. License scope is recorded in [LICENSE.md](../LICENSE.md); all imported source bytes are checked by `python3 scripts/verify_repository.py`.

| Directory | Origin and pin |
|---|---|
| `runtime/paradigma-inc-limite-violetto-cd68d27` | [Paradigma serving code](https://github.com/paradigma-inc/limite-violetto/tree/cd68d27f100ebb7c84f05603d5af0e8f934b28b5), complete retained 17-file snapshot |
| `model` | [Violetto model](https://huggingface.co/paradigma-inc/limite-1b-violetto/tree/9402e422e87b220507963cd42c997459c1e87b43), tokenizer/config/card/assets only; weights and upstream LFS attributes excluded |
| `value-model` | [Value-model backbone](https://huggingface.co/paradigma-inc/limite-1b-value-model/tree/ce83740e0e853a7d4e5b80fca29a65fd5b8db7fb), small source/config/card subset; no weights |
| `torch-2.7-linear` | PyTorch `134179474539648ba7dee1317959529fbd0e7f89`, selected arithmetic sources and installed-header snapshots |
| `sleef-pinned` | SLEEF `56e1f79cb140fb9326d612d0be06b5250565cade`, the pinned Torch submodule |
| `coremltools-9.0-bridge` | Apple coremltools 9.0 inspected bridge files, identified by the [recorded source hashes](../evidence/04b-reliability/coremltools-9.0-bridge-hashes.json) |

The import verifier checks these bytes against both the export inventory and retained upstream/source identity records. The model tokenizer is approximately 11.4 MB and is stored directly in Git; this repository does not require Git LFS.

Ancillary autoresearch, Lean, coremltools-main excerpts, and the redundant runtime tarball remain local. Their omission is intentional. Do not run `acquire_reference.py` or `verify_sources.py` merely to check a clone: they address the fuller historical acquisition and may overwrite evidence.
