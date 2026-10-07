# License scope

This import preserves existing upstream licenses and attribution. It does not assign a new blanket license or claim ownership of third-party material. No additional project-wide license grant for original research glue or reports is introduced by this cleanup; the owner can choose that policy separately.

| Material | Retained license / basis |
|---|---|
| Paradigma Violetto serving snapshot | [Apache License 2.0](sources/runtime/paradigma-inc-limite-violetto-cd68d27/LICENSE), declared in its README and package metadata |
| Violetto model assets and excluded weights | Apache-2.0 declared by the [pinned model card](sources/model/README.md); no separate model LICENSE file was provided in that snapshot |
| Paradigma value-model snapshot | Apache-2.0 declared by its [pinned model card](sources/value-model/README.md); this includes the backbone reused by the portable implementation |
| PyTorch source excerpts | [Retained PyTorch license](sources/torch-2.7-linear/LICENSE), including component terms; the adapted GGML reduction notice is reproduced in [coreml_port/NOTICE.md](coreml_port/NOTICE.md) |
| SLEEF source excerpts | [Boost Software License 1.0](sources/sleef-pinned/LICENSE.txt) |
| Apple coremltools 9.0 bridge excerpts | [Retained BSD-3-Clause license](sources/coremltools-9.0-bridge/LICENSE.txt) |

The upstream snapshots are unmodified except that the model repository's Git LFS attributes are intentionally not imported. Derived implementations and their modification scope are described in [NOTICE.md](NOTICE.md) and the existing [Core ML notices](coreml_port/NOTICE.md). Third-party licenses continue to govern their respective material.

Ancillary autoresearch, Lean, and coremltools-main snapshots remain in the local archive because their retained provenance/license package was incomplete for this export. They are not silently covered by the licenses above. No weights, installed dependency wheels, compiled model packages, or native binaries are distributed by this repository.
