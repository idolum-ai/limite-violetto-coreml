# Step 1 — Source and reference investigation

Date: 2026-09-22. Investigation complete; independent numerical baseline unavailable.

**Recommendation: qualify and reuse Paradigma's published value-model backbone, then proceed to the representative Core ML component experiment. Keep original-model fidelity unverified until reference outputs are obtained.** Daniel's suggestion to search the other repositories uncovered portable upstream code: 59 configuration fields match and all 743 expected parameter entries align after explicit scalar mapping. [Expanded search and compatibility evidence](EXPANDED-REFERENCE-SEARCH.md). No model inference or conversion has run.

## Gate results

| Gate | Result | Evidence and limitation |
|---|---|---|
| G1.1 Reproducible inputs | PASS for Step 1 assets | Immutable revisions captured; 17 runtime files and 9 model assets match upstream identities. Full checkpoint deliberately not downloaded; its upstream size/hash are recorded, not locally verified. |
| G1.2 Reference audit | PASS | Expanded to all 3 listed public GitHub repos and all 4 Hugging Face model inventories. Found a portable upstream backbone and critic example scores; no Violetto neural-output captures. |
| G1.3 Independent baseline | BLOCKED | No attributable neural-output fixtures in the inspected distribution. Original training/oracle files mentioned in comments are absent. No supported NVIDIA run was performed. |
| G1.4 Environment selection | PASS for installation, imports, and contract tests | CPython 3.12.9, torch 2.7.0, coremltools 9.0, transformers 5.6.2; 42 dependencies hash-pinned. Conversion compatibility is still untested. |
| G1.5 Next-step decision | PASS | [Capture specification and next experiment](BASELINE-CAPTURE.md) define how to close the baseline gap and what to test next. |

## Source inventory

- Runtime: `paradigma-inc/limite-violetto`, commit `cd68d27f100ebb7c84f05603d5af0e8f934b28b5`. [Commit metadata](runtime-commit.json), [complete Git tree](runtime-tree.json), [downloaded source](../../sources/runtime/paradigma-inc-limite-violetto-cd68d27/README.md).
- Model: `paradigma-inc/limite-1b-violetto`, revision `9402e422e87b220507963cd42c997459c1e87b43`. [Repository metadata](model-metadata.json), [configuration](../../sources/model/config.json), [model card](../../sources/model/README.md).
- [Challenge snapshot](challenge-0.md), [acquisition log](acquisition.log), [acquisition URLs and hashes](acquisition.json), [verified source identities](source-integrity.json), and [manifest](../../manifest.json).
- Downloaded tokenizer, tokenizer configuration, special-token map, chat template, generation configuration, model card, and supporting small assets. No weights downloaded.
- Expected checkpoint: `model.safetensors`, **2,070,811,520 bytes**, upstream LFS SHA-256 `9dc7e89e30acf473629f98ac3ade535fb05cc044fe7e399b78d0bc957283a5f5`. This is upstream metadata, not a locally calculated checkpoint hash.
- Serving code includes an [Apache-2.0 license](../../sources/runtime/paradigma-inc-limite-violetto-cd68d27/LICENSE). The pinned model card declares Apache-2.0 for weights; the model repository has no standalone LICENSE file. Both forms of provenance are retained.

Actual machine observation differs from the challenge: **Apple M4, 32 GiB unified memory, 10 physical/logical CPUs, macOS 26.5.2 (25F84), arm64**. [Machine record](machine.json). The challenge listed M4 Pro / 48 GB / macOS 26.6.2. Future performance estimates must use the observed target. Device form factor was not checked.

Inference from configuration, not a memory measurement: a full preallocated FP16/BF16 K+V cache at 2,048 tokens, batch one, 48 layers, 2 KV heads, and head dimension 128 occupies **96 MiB**. This excludes weights, activations, attention temporaries, conversion/compiler memory, and runtime copies. [Calculation and input validation](local-input-validation.json).

## Reference audit

The pinned runtime has 17 files, including seven runtime Python files and one test file. There are **10 test functions yielding 12 parametrized cases**. [Inventory](reference-inventory.json).

The published [contract tests](../../sources/runtime/paradigma-inc-limite-violetto-cd68d27/tests/test_package_contract.py) check package contents, dependency/registration contracts, window/configuration handling, and repeated synthetic weight loading with scalar folding. Their attention class is a stub with no forward implementation. They do not execute Violetto attention, generation, or the real checkpoint.

**Observed result: 12 passed in 5.45 seconds.** [Test log](upstream-contract-tests.log), [JUnit results](upstream-contract-tests.xml). These ran against the pinned source in the local inspection environment, not upstream's locked CUDA environment. No vLLM installation or NVIDIA serving check was performed.

Implementation comments refer to `limite_model.py`, `spec.py`, and `tests/test_limite_vllm_model.py`; none is in the pinned tree. The downloaded model repository contains no reference implementation or numerical tensor fixtures, and its card says direct Transformers support is forthcoming. The runtime's [published releases list](runtime-releases.json) was empty at inspection time. This finding is limited to those inspected distributions; it does not establish that no reference exists privately or elsewhere.

## Baseline status

We have source-level descriptions and passing package tests, **not an independent neural-output baseline**. Upstream benchmark tables and source comments about oracle comparisons do not supply the exact token inputs, tensors, execution configuration, and numerical outputs required for this port.

The expanded search found a [portable backbone and published critic smoke-check scores](EXPANDED-REFERENCE-SEARCH.md) in another Paradigma model repository. This supplies a promising implementation to qualify and reuse; it does not yet supply Violetto's next-token outputs. The practical numerical baseline target remains the supported public vLLM runtime at the exact pinned revisions on Linux/NVIDIA. That would validate fidelity to the serving implementation. Agreement with the separately mentioned training/oracle implementation would remain an upstream claim unless its code or captures become available.

[BASELINE-CAPTURE.md](BASELINE-CAPTURE.md) specifies required provenance, a fixed case matrix, selected intermediate tensors, forced continuations, repeatability runs, and an acceptance procedure. Numeric tolerances are intentionally **unset** until baseline variation is measured. No maintainer messages or infrastructure purchases were made.

## Environment

| Component | Pinned selection | Basis |
|---|---|---|
| CPython | 3.12.9 arm64 | Already installed; matches upstream's Python 3.12 requirement and available Core ML wheels |
| coremltools | 9.0 | Official release; macOS arm64 CPython 3.12 wheel available |
| torch | 2.7.0 | Core ML Tools 9.0 release notes support PyTorch 2.7; its dependency declaration caps tested Torch at 2.7.0 |
| transformers | 5.6.2 | Match the runtime's tokenizer/configuration version |
| numpy | 2.2.6 | Pinned compatible wheel; imports and checks succeeded |
| pytest | 9.1.1 | Match upstream's test-tool pin |

Evidence: [Core ML 9.0 release](coremltools-9.0-release.json), [tag identity](coremltools-9.0-tag.json), [version declaration](coremltools-9.0-deps.py), [requirements](../../requirements-investigation.in), [full hash lock](../../requirements-investigation.lock), [resolution report](pip-resolution.json), [installation report](pip-installation.json), [dependency check](pip-check.log).

All packages are in `../../.venv`; package caches are in `../../.cache`. The upstream CUDA lock was retained as source evidence and was not installed. Its torch 2.11.0+cu130 stack differs from this environment, so contract-test success here does not certify upstream numerical behavior or the eventual converted graph.

`coremltools`, `torch`, and `transformers` imported successfully. Metal/MPS availability was **true outside the sandbox**; the same availability check returned false inside it. [Host input/package inspection](local-input-validation.json), [sandbox result](local-input-validation-sandbox.json). No GPU compute or Core ML prediction was performed.

The general Apple installation page still refers to older releases, so version selection used the specific 9.0 release metadata and source declaration instead. `uv` dependency resolution encountered network routing errors; Python `pip` succeeded, and all selected distributions were then hash-pinned and installed with `--require-hashes`. [Failed uv resolution](environment-resolve.log), [successful pip installation](pip-installation.log).

## Input and numerical details established

The published config passes upstream validation. Its serialized window distance of 1,024 becomes a runtime span of 1,025. It specifies 48 layers; local/global attention; two mixer locations at zero-based layers 24 and 47, each with attention-input and residual-base mixing; BF16 positional tables; and `oracle_exact` head precision.

**Correction from Step 2:** 0.0078125 and 0.0009765625 are the BF16/FP16 *dtype machine epsilons*, recorded by the Step 1 inspection; they are not the effective default RMS epsilon. Local numerical probes and the pinned PyTorch 2.11 composite/CUDA source show that BF16/FP16 RMS normalization uses **FP32 opmath epsilon, 1.1920928955078125e-7**. The original interpretation repeated an incorrect upstream comment. Preserve the actual operation explicitly. [Correction and evidence](../02-components/REPORT.md#critical-numerical-correction). The source also requires post-value-embedding V in the cache and ordered mixer accumulation.

The tokenizer loaded the separate `chat_template.jinja` correctly. Three canonical math prompts were formatted and tokenized; token counts were **44, 58, and 48**. Their exact IDs are saved in [local-input-validation.json](local-input-validation.json). Each round-tripped to the exact formatted prompt, and the Transformers path agreed with direct loading of the published `tokenizer.json` on these cases. These are prepared inputs, not generated answers.

Two integration details matter:

1. Tokenizer metadata EOS is `151643` (`<|endoftext|>`), while the published model and generation config use `151645` (`<|im_end|>`). Use the published generation configuration for stopping rather than substituting `tokenizer.eos_token_id`.
2. Transformers 5.6.2 emitted a Mistral-regex warning when loading this local Qwen2 tokenizer. Inspection of its installed source shows a broad local-config detection path when the model config lacks `transformers_version`. We did not apply the suggested regex rewrite: preserving the published tokenizer is the relevant starting point, and direct-tokenizer agreement passed on the three cases. This does not establish agreement on every possible input or with a future upstream capture. The warning remains in [the log](local-input-validation.log); [source excerpt](tokenizer-warning-source.txt) records the detection logic.

The first inspection script attempts assumed an embedded chat template and a list return value from `apply_chat_template`. The published assets store the template separately, and this Transformers version returns `BatchEncoding` by default. The inspection script was corrected to check the loaded template and explicitly request/read the dictionary result. Earlier failures are preserved in [attempt 1](local-input-validation-attempt1.log) and [attempt 2](local-input-validation-attempt2.log). No source/model assets were edited to make the checks pass.

## Decision

Proceed next by qualifying the discovered upstream backbone and using its eager attention/mixer code for **Step 2's stateful component experiment**, with explicit precision/masking/cache tests. Its 743-entry schema match is encouraging but does not replace a strict real-checkpoint load. Start small, then test production attention dimensions. The current evidence supports this bounded experiment, not a claim that the full model will convert or run quickly.

Closing G1.3 requires attributable upstream captures or a supported NVIDIA reference run. That dependency need not prevent component feasibility work, but must remain visible in every fidelity claim. Steps 2–5 are not executed in this session.

Reproduction commands are in [README.md](../../README.md); all gates and future evidence slots are in [PLAN.md](../../PLAN.md).
