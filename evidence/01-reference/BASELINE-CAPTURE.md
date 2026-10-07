# Independent baseline capture specification

Status: specification only; no model outputs captured.

Update: a portable upstream backbone was found in `paradigma-inc/limite-1b-value-model`. See the [expanded audit](EXPANDED-REFERENCE-SEARCH.md). It provides reusable reference code and approximate critic smoke-check values, but no Violetto token-score baseline. Qualify it before implementing a separate portable forward calculation.

## Reference identity

Preferred available execution target: the public supported vLLM implementation at runtime revision `cd68d27f100ebb7c84f05603d5af0e8f934b28b5`, with model revision `9402e422e87b220507963cd42c997459c1e87b43`. This validates agreement with the supported serving implementation. It does not independently prove agreement between that implementation and the unpublished training/oracle code mentioned in source comments.

Use Linux x86-64, Python 3.12, upstream's locked vLLM 0.26.0 / torch 2.11.0+cu130 / transformers 5.6.2 environment, and a compatible NVIDIA GPU/driver. Record exact Python patch version, `uv.lock` hash, dependency inventory, driver/CUDA versions, attention backend, compile mode, dtype, cache dtype, device, seeds, and all execution settings. Set tensor/pipeline parallel size to one. Record the actual `lm_head_precision_mode=oracle_exact` config rather than assuming an FP32 projection: the source computes the projection at activation precision in this mode and applies the sigmoid score transformation in FP32.

Acquire reference outputs by either:

1. Maintainer-provided captures with this provenance and the inputs below; obtaining them requires an authorized request.
2. An authorized run of the unmodified supported implementation on available NVIDIA hardware, with separately recorded instrumentation for intermediate captures. No cloud run or correspondence has been initiated.

## Fixed cases to prepare before candidate evaluation

| Case | Inputs | Purpose |
|---|---|---|
| Arithmetic | Canonical chat for `Compute 17 * 23.` | Short real prompt; answer 391 |
| Multi-step | Canonical chat for `Solve x^2 - 5x + 6 = 0 and give the sum of its roots.` | Short multi-step prompt; roots 2, 3 and sum 5 |
| Equation | Canonical chat for `Find all integer solutions of x^2 = 49.` | Multiple-solution case; -7 and 7 |
| Shape matrix | Fixed token sequences of lengths 1, 2, 17, 128 | Positions, masking, prefill/decode shape handling |
| Window matrix | Fixed token sequences of lengths 1024, 1025, 1026, 2048 | At zero-based q=1024, key 0 is visible; at q=1025, key 0 is excluded and key 1 remains visible in local layers |
| Chunk matrix | Same tokens passed as full prefill, token-by-token, and several chunk splits | Cache equivalence and chunk-position accounting |
| Decode | At least 8 forced continuation steps plus a separate greedy continuation | Compare on identical prefixes before autoregressive divergence |
| Reset | A, then B with a new state, compared with B alone | Request isolation |

Freeze literal token arrays and their hashes before model comparison. The local tokenizer inspection can prepare inputs, but cannot supply expected neural outputs. Both mixer layers (zero-based 24 and 47) run on every complete forward pass; instrument their attention-input and residual-base paths.

## Required artifacts per case

- `metadata.json`: revisions, config and weight hash, command, environment, execution mode, and capture provenance.
- `input.json`: exact raw messages, formatted prompt, token IDs, positions, valid lengths, mask convention, explicit sampling/stop settings, and fixed continuation tokens.
- Tensor files in a documented non-executable format (for example safetensors or NPZ loaded with `allow_pickle=False`): full next-token scores for each forced step, and selected intermediates.
- Selected intermediates: initial normalized embeddings; local/global Q/K and post-value-embedding V; attention outputs; mixer inputs/coefficients/outputs at layers 24 and 47; final normalized hidden state; projection before and scores after output softcapping. Record dtype and shape for every tensor.
- `continuation.json`: greedy tokens and stopping reason; canonical sampled outputs are a separate task-level observation.
- `repeatability.json`: at least three repeat runs of the baseline and prefill/cached comparisons to characterize its own variation.

## Acceptance procedure

1. First verify artifact identity, tensor shapes, input equivalence, and finite outputs.
2. Measure baseline repeatability and known execution-path variation. Record max/mean absolute error, relative error with an explicitly defined denominator floor, and next-token top-1/top-k agreement.
3. Freeze per-tensor error limits and ranking/task criteria in a versioned acceptance file **before** evaluating the candidate. Exact values remain unset until baseline measurements exist.
4. Compare forced continuations so a changed greedy token does not invalidate all later tensor comparisons.
5. Inspect first divergence, near-tie margins, and completed math answers. Short task success alone does not establish numerical fidelity.
6. Separately label original-runtime comparison, portable/cache self-consistency, and Core ML/portable comparison. Passing the latter two cannot substitute for the first.

## First conversion experiment after Step 1

Use the pinned local environment to test a small stateful attention component with preallocated K/V buffers and explicit masks. Start with deterministic toy weights at small dimensions, then use production attention dimensions (10 query heads, 2 KV heads, head dimension 128). Include the post-VE write and global/local mask variants. Isolate ordered mixer/precision tests so errors can be diagnosed independently. Verify repeated updates and a fresh state before integrating components. These measurements are conversion feasibility and internal consistency evidence; model fidelity remains unverified while this baseline is absent.
