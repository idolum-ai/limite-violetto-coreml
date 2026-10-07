# Model architecture and training

For installation, examples and output interpretation, see the [usage guide](README.md).

## Architecture

| Property | Value |
|---|---|
| Parameters | 1,035,255,457 |
| Backbone | GMPT transformer, 48 layers, hidden size 1,280 |
| Attention | 10 query heads, 2 key/value heads. 36 layers use sliding-window attention with a window of **1,024 tokens** (inclusive of the query token); the remaining 12 layers (indices 3, 7, 11, ..., 47) attend to the whole prefix |
| Positional encoding | Rotary positional embeddings (RoPE) on local-attention layers |
| Additional components | Token-dependent value embeddings, cross-head attention correction (XSA), dynamic dense residual mixing (MUDD), learned layer scales, a learned per-head attention gate applied as `2 * sigmoid(...)` before `o_proj` (128 gate channels), and QK RMS normalisation applied before RoPE |
| Value head | Linear projection from 1,280 hidden features to one scalar, with a learned bias |
| Maximum context | 131,072 tokens, including the prompt |
| Weight storage | FP32 |
| Inference precision | BF16 backbone, FP32 value head |

The weights contain the full trained backbone and value head. The supplied loader restores both components.

## Training objective

The model was initialized from a supervised fine-tuned language-model backbone, fitted offline to correctness returns, and subsequently updated during online actor–critic training.

At each response action position, the model predicts the final binary correctness return using only the preceding tokens. Training minimizes half the mean squared error over valid response tokens, with discount factor 1, critic lambda 1 and no length penalty.

During the online phase, the critic used Adam with learning rate `5e-6`, betas `(0.9, 0.95)`, epsilon `1e-8`, weight decay `0` and gradient-norm clipping at `1`.

## Scope and limitations

The training domain is mathematical reasoning. The output is a raw value estimate and is not constrained to `[0, 1]`. It is not a calibrated probability or a proof of correctness.

Online training used informative response groups containing both correct and incorrect answers. Calibration on this distribution need not transfer to unrestricted generations. Ranking quality can also change with the generator, sampling settings, domain or reasoning style.

Evaluate candidate-ranking rules and decision thresholds on labeled examples from your intended application. A value before a response token does not assess that token's correctness: the token has not yet been observed.

## Inference API

Use `load_value_model`, `ValueModel.forward`, `ValueModel.score_tokens` or `score.py` as described in the [usage guide](README.md). The API supports one unpadded sequence at a time. It does not implement padded batches, a KV cache or text generation.
