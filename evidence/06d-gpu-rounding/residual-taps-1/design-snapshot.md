# Round-fixed group-0 residual localization

This is a side-by-side, output-only instrumentation of `artifacts/06d-gpu-rounding/group0-1/0.mlpackage`. Its pinned completed build record SHA is `4e6ac1baa2f70c10bf812523f7e341d9dabf34b30299e4819631f2db31cf150f`. It is the original group0 with all131 MIL rounds replaced by the qualified v3 experiment, not the conditional CPU candidate.

Seven existing SSA values bracket layer0 query/attention and the input to layer1 keys:

| Value | Stage | Original index | Candidate index | FP32 shape |
| --- | --- | ---: | ---: | --- |
| q_3 | layer0 final rotary query | 1679 | 1922 | [1,Q,10,128] |
| scores_1 | selected scaled attention scores, before causal masking | 4380 | 4704 | [1,10,R,2048] |
| weights_1 | softmax weights | 7216 | 7567 | [1,10,R,2048] |
| x_37 | raw attention value product, sliced to Q, before post-product Q | 7349 | 7700 | [1,Q,10,128] |
| value_19 | layer0 final hidden | 10698 | 11589 | [1,Q,1280] |
| x_93 | layer1 normalized projection input | 11170 | 12088 | [1,Q,1280] |
| x_97 | layer1 raw key projection before Q | 11776 | 12721 | [1,Q,256] |

Q uses axis1 with default3 and range1..32. Score/weight R uses axis2 with default32 and range1..32, allowing prepared/padded query rows. Their entire tensors are saved/compared; later causal interpretation distinguishes the first actual Q rows from padded rows. `scores_1` precedes the `x_33` causal-mask select, so it does not intentionally include masked infinities. Finite values are a runtime check, not inferred from the name alone.

The builder validates every tap name, original/candidate index, operation type, tensor type/shape, dependency chain, and producer bytes against the original sealed group0. It pins the candidate package's complete file inventory to the completed v3 build record. All46,035 candidate operations remain byte-identical; only the description output list and main block output list gain entries. Removing those entries must reconstruct the entire original candidate protobuf exactly. Original outputs (`y`,`tap23_out`), all8 state interfaces/operations, input shapes, all weights, and package metadata remain unchanged. The package inventory must differ only at its model protobuf.

Adding outputs can change native optimization despite identical operation bytes. Diagnostic validity therefore requires instrumented output `y`, `tap23_out`, and every element of all8 post-states to match the untapped candidate byte-for-byte **separately under CPU_ONLY and CPU_AND_GPU** on both recorded chunks. The root-owned replay worker performs these controls; a build's structural pass alone is not a diagnostic pass. Compare candidate CPU versus candidate GPU taps only after their same-policy controls pass. Original-baseline matched tap packages are not needed for this first bounded comparison.

Build CLI for root only:

```sh
.venv/bin/python scripts/build_rounding_residual_taps.py --attempt residual-taps-1 --compile
```

Optional compilation installs the existing guards and requires unchanged prediction counters. The builder never predicts. No full-model expansion, arithmetic changes, extra internal operations, or score-policy changes are part of this task.
