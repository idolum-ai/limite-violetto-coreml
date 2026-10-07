---
license: apache-2.0
language:
- en
tags:
- value-model
- critic
- reinforcement-learning
- math
- custom-code
---

# Limite Value Model

A 1.035B-parameter value model for mathematical reasoning. Given a question and a proposed solution, it returns one value per response token. Each value estimates the final correctness return from the reasoning available *before* that token.

**Values are regression scores, not calibrated probabilities of correctness.** The model scores text. It does not generate answers and it does not verify proofs.

The repository contains everything needed for inference: weights, tokenizer, configuration and Python code. No separate actor model or training framework is required.

## Install and download

Use Python 3.12. CUDA inference was tested on an NVIDIA H100; short examples also work on CPU. Download or clone this Hugging Face repository, including its Python files and tokenizer, rather than only the weight file.

```bash
python -m pip install huggingface-hub
hf download paradigma-inc/limite-1b-value-model --local-dir ./limite-1b-value-model
cd ./limite-1b-value-model
python -m pip install -r requirements.txt
```

To clone instead, install Git LFS and run `git lfs install` before `git clone https://huggingface.co/paradigma-inc/limite-1b-value-model`; otherwise, `model.safetensors` may be a small pointer file instead of the actual weights. `hf download` does not require Git LFS.

Run the remaining commands in this guide from the downloaded repository's root directory.

## Score your first solution

```bash
python score.py \
  --model-dir . \
  --device cuda \
  --attention-backend eager \
  --prompt 'How many positive divisors does 60 have?

Please reason step by step, and put your final answer within \boxed{}.' \
  --response 'The prime factorisation is 60 = 2^2 * 3 * 5, so the number of divisors is (2+1)(1+1)(1+1) = 12. The answer is \boxed{12}.' \
  --output scores.json
```

The command prints the scores and saves them to `scores.json`. It requires no FlashAttention installation. For CPU, replace `--device cuda` with `--device cpu`.

The reference scores and timings below were measured on an **NVIDIA H100 80GB**, with **Python 3.12**, **`torch==2.11.0`** and **CUDA 13.0**, using a single unpadded sequence. Timings are the mean of **five calls after warmup**.

For this exact example, the `eager` backend returns approximately:

```text
prompt_value:                      0.696658
mean_response_pre_action_value:    0.680541
```

The two FlashAttention backends return approximately:

| Backend | `prompt_value` | `mean_response_pre_action_value` |
|---|---|---|
| `flash_attention_2` | 0.696323 | 0.680799 |
| `flash_attention_3` | 0.696323 | 0.680787 |

Small differences across hardware and attention backends are expected. On this **100-token example** (46 prompt + 54 response), in the environment above, the three backends agree within **2.2e-3 per token**. Deviations grow with sequence length: the maximum difference between FA2 and FA3 was **5.6e-3** on a **6,064-token input**. These are measurements on specific inputs, not universal tolerances for diagnosing a build or thresholds for accepting an answer as correct.

### The prompt format the model was trained on

Training prompts place the question in a `user` turn followed by a blank line and this exact instruction:

```text
Please reason step by step, and put your final answer within \boxed{}.
```

Responses begin with a `<think>` reasoning span and end with `<|endoftext|>`. Scores on inputs far from this format are outside the distribution the model was fitted on.

## Use it from Python

Run this from inside the downloaded repository. Load the model once and reuse it for multiple solutions:

```python
import torch
from value_model import load_value_model, tokenize_pair

model = load_value_model(
    ".",
    device="cuda",              # or "cpu" for short inputs
    attention_backend="eager",
)

prompt_ids, response_ids = tokenize_pair(
    ".",
    prompt="How many positive divisors does 60 have?\n\nPlease reason step by step, and put your final answer within \\boxed{}.",
    response=r"The prime factorisation is 60 = 2^2 * 3 * 5, so the number of divisors is (2+1)(1+1)(1+1) = 12. The answer is \boxed{12}.",
)
scores = model.score_tokens(prompt_ids, response_ids)

print("Value before the response:", scores["prompt_value"])
print("One value per response token:", scores["pre_action_values"])
print("Mean pre-action value:", scores["mean_response_pre_action_value"])
```

For several candidates, call `tokenize_pair` and `model.score_tokens` for each candidate using the same loaded model. The reference API accepts **one unpadded sequence at a time**. It does not implement padded batches, a KV cache, or text generation. Use the supplied loader rather than `AutoModelForCausalLM` or `.generate()`.

## Read the scores correctly

| Field | Meaning |
|---|---|
| `pre_action_values` | One value per response token, based on the prefix **before** that token |
| `prompt_value` | Value before any response token; it depends on the prompt and cannot rank different solutions to the same prompt |
| `mean_response_pre_action_value` | Arithmetic mean of the response's prefix values; a possible aggregation for analysis or candidate ranking |
| `last_pre_action_value` | Value before the last response token |
| `raw_value_after_complete_input` | Value after the entire input; a different quantity from the pre-action values |

**Do not apply a sigmoid.** The model was trained to regress raw returns. Its values can be outside `[0, 1]`, and the loader does not clip them. In particular, a mean value of `0.8` does **not** mean an answer has an 80% chance of being correct. Validate any ranking rule or threshold on your own labeled examples.

Each pre-action value sees only the prompt and earlier response tokens. It does not see the token being scored or the rest of the solution. Consequently, an incorrect step can affect later values, but its own pre-action value is not an assessment of that step's correctness.

## Score long solutions

The default `eager` backend is intended for short examples. Its attention memory grows quadratically and the loader rejects inputs longer than **4,096 tokens, including the prompt**.

For longer solutions, install a compatible FlashAttention backend and select it explicitly:

```bash
python score.py \
  --model-dir . \
  --device cuda \
  --attention-backend flash_attention_2 \
  --tokens-json rollout.json --output scores.json
```

The two FlashAttention backends are alternatives with different requirements. To select FA3, use `--attention-backend flash_attention_3` in the command above.

| Backend | Requirements | Verified build |
|---|---|---|
| `eager` | Dependencies in `requirements.txt`; CPU or CUDA; at most 4,096 input tokens | `torch==2.11.0` |
| `flash_attention_2` | Compatible CUDA GPU and `flash-attn` installation | `flash-attn==2.8.3.post1` |
| `flash_attention_3` | Compatible Hopper GPU and FA3 build exposing `flash_attn_interface` | `flash_attn_3` 3.0.0, built from `hopper/` at [`edb5c76ee329b18ed95d1f7ea9aa522a1331ab7d`](https://github.com/Dao-AILab/flash-attention/tree/edb5c76ee329b18ed95d1f7ea9aa522a1331ab7d/hopper) |

See the upstream [FlashAttention installation instructions](https://github.com/Dao-AILab/flash-attention#installation-and-features) and [separate Hopper build for FA3](https://github.com/Dao-AILab/flash-attention#flashattention-3-beta-release). The commands below reproduce the verified builds in the environment above. The tested FA3 build exposes the top-level `flash_attn_interface` module; installing FA2 alone does not provide it. CUDA extensions are not bundled here.

The build tools locate the CUDA toolkit through `CUDA_HOME` or `nvcc` on your `PATH`. If needed, set `CUDA_HOME` to **your own CUDA installation directory**, containing `bin/nvcc`; no fixed installation path is required.

The FA2 command below targets **Hopper/H100**: `FLASH_ATTN_CUDA_ARCHS=90` selects `sm90`. For **A100**, use `FLASH_ATTN_CUDA_ARCHS=80` (`sm80`). Omit the variable to build the package's default set of architectures supported by your CUDA toolkit, at the cost of substantially longer compilation. FA3's separate Hopper build does not support A100.

```bash
# FA2 on H100 (Hopper)
pip install ninja
FLASH_ATTN_CUDA_ARCHS=90 MAX_JOBS=32 \
  pip install flash-attn==2.8.3.post1 --no-build-isolation
```

```bash
# FA3 (separate Hopper build)
pip install ninja
git clone https://github.com/Dao-AILab/flash-attention.git
git -C flash-attention checkout edb5c76ee329b18ed95d1f7ea9aa522a1331ab7d
cd flash-attention/hopper
MAX_JOBS=32 pip install . --no-build-isolation
cd ../..
```

**Build time:** `ninja` is required for parallel compilation. Without it, compilation is serial and can take hours instead of minutes. This FA3 build has no prebuilt wheels and must be compiled from source. With 32 CPU cores, the measured build times were approximately **10 minutes for FA2** (one CUDA architecture) and **45 minutes for FA3**.

Measured scoring latency in the environment above, for a single unpadded sequence. The two shorter columns are the mean of five calls after warmup; the two longer columns are a single call:

| Backend | 44 tokens | 6,064 tokens | 103,996 tokens | 131,072 tokens |
|---|---:|---:|---:|---:|
| `eager` | 53.0 ms | Not supported | Not supported | Not supported |
| `flash_attention_2` | 42.1 ms | 59.9 ms | 2.13 s | 2.49 s |
| `flash_attention_3` | 44.6 ms | 55.1 ms | 1.71 s | 1.73 s |

The configured context limit is **131,072 tokens including the prompt**. No input is silently truncated. The weight file is approximately **4.14 GB**, stored in FP32. Inference uses a BF16 backbone and FP32 scalar head. With FlashAttention, peak GPU memory was **6.3 GiB** at 103,996 tokens and **7.4 GiB** at the full 131,072-token context, so the whole context fits well inside a 24 GB card.

## Use original rollout tokens

If you already have tokenized rollouts, use the original prompt and response token IDs to preserve their exact boundary. Save a JSON object with two integer lists named `prompt_ids` and `response_ids`, then pass its path with `--tokens-json` as above. The Python equivalent is `model.score_tokens(prompt_ids, response_ids)`.

Both lists must be nonempty and contain no padding, and must be plain Python `list` or `tuple` objects of integers: `torch.Tensor` and NumPy arrays are rejected. Every ID must lie in `[0, 151667)`. The embedding matrix has 151,680 rows, but the rows above 151,666 are padding rather than tokens, and an ID in that range raises.

The text helper applies the included chat template to the prompt and tokenizes the response with `add_special_tokens=False`. It does not append EOS.

**End of sequence.** A completed rollout ends with `<|endoftext|>` (ID 151643). `<|im_end|>` (ID 151645) closes the turns of the prompt, not the generation. Include the trailing `<|endoftext|>` when you want to score the sequence exactly as it was produced.

## Token alignment for RL integrations

The convenience API returns:

```text
pre_action_values[t] = V(prompt, response[:t])
```

The low-level forward returns **post-token** values with shape `[1, sequence_length]`. Convert them to pre-action values using this offset:

```python
input_ids = torch.tensor([prompt_ids + response_ids], dtype=torch.long,
                         device=model.device)
post = model(input_ids)
P, L = len(prompt_ids), len(response_ids)

# Exactly one baseline BEFORE each response action, shape [1, L]:
pre_action_values_2d = post[:, P - 1 : P + L - 1]

# The flat form the contract above indexes, shape [L]:
pre_action_values = pre_action_values_2d[0]

# This has seen the final token and is a different quantity:
raw_value_after_complete_input = post[:, -1]
```

Using `post[:, P:P+L]` for the same actions causes an off-by-one error: those values have already seen the corresponding action. In an RL return calculation, a truly terminal transition has a zero bootstrap; do not replace it with `raw_value_after_complete_input`.

## Model details and limitations

It was trained on mathematical reasoning and binary final-correctness returns. Calibration and ranking quality can change with the generator, sampling settings, domain or style of reasoning. It is not a symbolic verifier or a general-purpose safety judge.

- [Architecture, training and limitations](MODEL_DETAILS.md)
