# Expanded Paradigma reference search

Date: 2026-09-22. Requested by Daniel after the initial runtime/Violetto-only audit.

**Found an upstream portable backbone in `paradigma-inc/limite-1b-value-model`. This materially improves the implementation path. It is a candidate reference implementation, not yet a numerical baseline for Violetto.**

## Coverage

Public GitHub organization inventory: **3 repositories**. [Inventory](paradigma-github-repos.json). Default-branch commits were pinned and recursive tree responses were not truncated.

| Repository | Inspected revision | Finding |
|---|---|---|
| `limite-violetto` | `cd68d27f100ebb7c84f05603d5af0e8f934b28b5` | The same 17-file runtime already inspected; no portable oracle or inference fixtures |
| `autoresearch` | `5a54379693231a9ed8bd22aba23ddb8e2aa4df3b` | Ten-file training experiment; read README and train.py. Different architecture: different rotary layout, post-rotary QK normalization, ReLU² MLP, and no MUDD implementation. Not the Violetto reference |
| `xi-log-concavity-lean` | `f807d86f642ec7a9404ef5d393dcd2c7b4ea17e0` | Inspected complete 9,276-file tree and README. Mathematical proof/formalization project; numerical certificates concern that theorem, not neural-model logits. Did not inspect every proof file's contents |

Public Hugging Face author inventories: **4 models, 0 datasets, 0 spaces** returned by the API. [Models](paradigma-hf-models.json), [datasets](paradigma-hf-datasets.json), [spaces](paradigma-hf-spaces.json). Each model's full file inventory was captured at its observed immutable revision.

| Model | Revision | Finding |
|---|---|---|
| `limite-1b-violetto` | `9402e422e87b220507963cd42c997459c1e87b43` | Weights/config/tokenizer assets; no reference code |
| `limite-1b-base` | `c55f6dd9741d89b88186235c6d43bff367ae3cb0` | Weights/config/tokenizer assets; no reference code or fixtures |
| `limite-1b-base-soup` | `9ade12f28483ec61896427bb983719501325e275` | Weights/config/tokenizer assets; no reference code or fixtures |
| `limite-1b-value-model` | `ce83740e0e853a7d4e5b80fca29a65fd5b8db7fb` | Portable `modeling_backbone.py`, `configuration_gmpt.py`, loader, and published example scores |

This covers the listed public repositories' current default revisions and model inventories. It is not an exhaustive search of all historical branches, private repositories, personal accounts, or Hub discussions.

## Portable backbone and observed compatibility

The value model publishes a PyTorch backbone with an eager CPU attention path, plus optional CUDA FlashAttention paths. It includes the local/global masks, BF16 rotary factors, value embeddings, XSA, learned scales, SwiGLU, and both MUDD paths needed by Violetto. [Backbone source](../../sources/value-model/modeling_backbone.py), [configuration](../../sources/value-model/configuration_gmpt.py), [published config](../../sources/value-model/backbone_config.json).

We compared the two published configs and constructed both models on PyTorch's **meta device** (shapes only; no real weights or inference):

- **59 shared config fields have identical values**, including dimensions, local/global layers, attention scale, rotary parameters, value embeddings, gates, XSA, MUDD topology, and MLP family.
- **743 backbone parameter tensors map to all 743 expected Violetto checkpoint source entries**, with no missing or unexpected names after mapping.
- The mapping includes **288 learned scalar parameters** (six per layer): rename the value backbone's `layers.N.scales.*` to Violetto's names and reshape `[1]` to scalar `[]` when comparing schemas. Adaptation into the backbone will apply the inverse shape change.
- Runtime fused QKV and gate/up projections were expanded into the source names that its loader accepts. This is a source-schema comparison, not inspection of the real checkpoint tensor header or values.

Evidence: [comparison JSON, including every parameter mapping](backbone-schema-comparison.json), [execution log](backbone-schema-comparison.log), [repeatable comparison script](../../scripts/compare_backbone_schema.py).

## What is and is not a baseline

The [value-model card](../../sources/value-model/README.md) gives one exact prompt/response example with approximate H100 eager outputs: `prompt_value ≈ 0.696658` and `mean_response_pre_action_value ≈ 0.680541`. It also records observed backend differences for specific sequences. Those are useful published **critic smoke-check results**, not per-layer fixtures or Violetto next-token scores. No value-model checkpoint was downloaded and those scores were not reproduced locally.

The critic was trained separately with a scalar value head. Its weights and scores cannot be substituted for Violetto's weights and generated-token scores. Its source can nevertheless provide a separately published implementation for comparison and a starting point for the portable port.

Remaining adaptation/qualification work:

1. Check real Violetto tensor names, shapes, and dtypes against the mapped source schema once its checkpoint is downloaded.
2. Preserve Violetto's mixed parameter precision. The value loader narrows every backbone parameter to BF16; Violetto retains selected parameters, including XSA alpha, in FP32. Blindly copying the value loader could change outputs.
3. Supply Violetto's tied embedding output projection and sigmoid score transformation at the declared `oracle_exact` precision. The value model has a different scalar head.
4. Preserve Violetto's canonical prompt and EOS. The critic's rollout format and EOS are different.
5. Add and validate cached generation; the published backbone only handles a full unpadded sequence starting at position zero.
6. Compare against supported Violetto-runtime captures before claiming fidelity. Shared provenance between upstream implementations also means agreement is useful evidence, not proof that every upstream assumption is correct.

The value model's architecture prose calls the local window “1,024 tokens inclusive”; its config and eager mask actually implement maximum distance 1,024, admitting 1,025 keys including the query. Use the explicit source/config convention and test the boundary.

## Updated recommendation

Qualify and reuse this upstream backbone before writing the full portable forward calculation from scratch. Start Step 2's component tests from its eager attention and mixer code, preserving source attribution and recording every necessary adaptation. The plan's independent Violetto-output gate remains unpassed, but the reference-code gap is substantially reduced.

Reproduce the schema comparison from the project root:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/compare_backbone_schema.py
```
