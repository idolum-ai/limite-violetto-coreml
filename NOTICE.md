# Attribution and local modifications

This research investigates **Paradigma Limite 1B – Violetto**, published by Paradigma:

- Model: https://huggingface.co/paradigma-inc/limite-1b-violetto at `9402e422e87b220507963cd42c997459c1e87b43`.
- Serving implementation: https://github.com/paradigma-inc/limite-violetto at `cd68d27f100ebb7c84f05603d5af0e8f934b28b5`.
- Reused value-model backbone: https://huggingface.co/paradigma-inc/limite-1b-value-model at `ce83740e0e853a7d4e5b80fca29a65fd5b8db7fb`.

`portable/` adapts the pinned backbone to the Violetto actor tensor/scalar layout, head, local CPU cache policy, and explicit stop handling. `coreml_port/` contains local Core ML graph transformations, ordered arithmetic, stateful execution, identity verification, and process-local guards. These are local research adaptations, not upstream releases or endorsements. `experiments/` and `scripts/` implement the local investigation and its validation.

The PyTorch/GGML, SLEEF, and Apple bridge derivations and notices are preserved in [coreml_port/NOTICE.md](coreml_port/NOTICE.md). Existing copyright and license text remains with each retained source family. The bridges are inspected source excerpts; no modified coremltools wheel is redistributed.

The October 7, 2026 cleanup adds navigation, archival boundaries, export verification, and staging checks. It redacts private filesystem paths and host-local battery identifiers from selected report copies, omits private trace entries from five exported checksum lists, and makes one synthetic unit-test fixture independent of an untracked native binary. Focused publication tests cover the new metadata exclusions. It does not modify the model/runtime arithmetic, thresholds, guards, or upstream code. See [import-manifest.json](docs/import-manifest.json) for original/export identities and [validation.md](docs/validation.md) for the checks actually rerun.
