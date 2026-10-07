# Violetto · Core ML research

A reproducible research record for running [Paradigma Limite 1B – Violetto](https://huggingface.co/paradigma-inc/limite-1b-violetto) on Apple Silicon: a portable PyTorch reference, a sharded Core ML implementation, and the experiments explaining their numerical differences.

**Recorded CPU comparisons pass on the pinned local host. GPU correctness remains unqualified.** This repository preserves the September 22–24, 2026 investigation; the October 7 cleanup did not rerun inference or certify a clean installation.

[Setup and reproduction](docs/setup.md) · [Evidence](evidence/README.md) · [Research roadmap](docs/roadmap.md) · [Archive and integrity](docs/archive.md) · [Licenses and attribution](LICENSE.md)

## What the evidence supports

| Path | Recorded result | Limits |
|---|---|---|
| Portable PyTorch CPU | Local full-prefix/cache checks pass; roughly 12–15 tokens/s in the earlier practical pilot | Original NVIDIA runtime parity is unavailable; MPS cache remains unqualified |
| Core ML CPU baseline | 54 short comparisons, 97 partition checkpoints, six capacity/window comparisons through 2,048 tokens, and a complete 674-token response pass | Comparison is against the qualified local reference |
| Promoted Core ML CPU | Conditional-score integration passes the original gates and all 674 response tokens; 17.3–18.1% lower decode latency in the bounded eight-trial comparison | Still substantially slower than the earlier portable CPU measurements; no universal speed claim |
| Persistent compiled CPU cache | Saved fresh-process reopen readiness of 13.9–14.3 seconds versus 240.4 seconds through package loading | Cache-conditioned startup, not a cold-machine measurement or a decode improvement |
| Core ML GPU policy | Full-model screen fails all 15 score comparisons; later probes isolate rounding and reciprocal differences | `CPU_AND_GPU` permits devices; actual placement is unverified; GPU is not adopted |
| Neural Engine / clean installation / production | Not qualified | No readiness claim |

The [CPU promotion decision](evidence/05g-integration/decision.json), [practical comparison](evidence/05-practical/REPORT.md), and [GPU residual report](evidence/06g-gpu-residual/REPORT.md) contain the supporting scope and caveats. Frozen acceptance criteria and historical failed attempts remain in the evidence export.

## Start here

For code and evidence review, only Python 3.12+ and Git are needed:

```sh
python3 scripts/verify_repository.py
```

For the recorded numerical environment, use **Apple M4, 32 GiB, macOS 26.5.2, CPython 3.12.9, Torch 2.7.0, and coremltools 9.0**. The [setup guide](docs/setup.md) explains the hash-pinned environment, checkpoint retrieval, bounded tests, and the native guards required before Core ML execution.

After installing the environment and verifying the checkpoint, the practical CPU entry point is:

```sh
.venv/bin/python -m portable.generate 'Compute 17 * 23.' \
  --device cpu --max-new-tokens 1024 \
  --sample --seed 0 --stop-at-tokenizer-eos
```

That explicit stop option adds tokenizer EOS 151643 to the published EOS 151645. Their discrepancy is [recorded](evidence/03-portable/eos-observation.json), not resolved. The total context budget is 2,048 tokens; length-limited output is incomplete.

**Core ML is an archive-dependent research path.** Models, native binaries, tensor fixtures, and host-bound compiled identity manifests are intentionally absent from Git. Use the [archive restoration instructions](docs/archive.md) and an explicit qualified candidate selection. The historical package-runner default selects an unqualified candidate. A clone alone is not a runnable compiled model distribution.

## Repository map

| Location | Purpose |
|---|---|
| [`portable/`](portable/) | Strict weight mapping, local-reference attention/cache behavior, generation CLI |
| [`coreml_port/`](coreml_port/) | Core ML graphs, ordered arithmetic, runtime, identity checks, native guard sources |
| [`experiments/`](experiments/) | Early component experiments and source qualification |
| [`scripts/`](scripts/README.md) | Conversion, diagnostics, gate aggregation, audits, and focused tests |
| [`sources/`](sources/README.md) | Pinned upstream snapshots with retained licenses; no weights |
| [`evidence/`](evidence/README.md) | Curated reports, numerical records, criteria, and historical checksum lists |
| [`docs/`](docs/) | Setup, archival boundaries, roadmap, and cleanup verification |
| [`manifest.json`](manifest.json) | Historical research manifest; read with the archive boundary notes |

The import preserves production/research Python sources byte-for-byte, apart from a documented synthetic-test fixture adjustment. It does not change model arithmetic, inference defaults, numerical tolerances, or qualified candidates. See [cleanup validation](docs/validation.md) for checks actually run during this import.
