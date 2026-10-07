# Output-only normalization localization

This fresh clone exposes the12 existing checkpoints proposed in `evidence/06d-gpu-rounding/next-experiment.json`. The source remains `artifacts/06d-gpu-rounding/group0-1/0.mlpackage`, pinned by build record SHA-256 `4e6ac1baa2f70c10bf812523f7e341d9dabf34b30299e4819631f2db31cf150f` and its complete package inventory. No prior script, artifact, runtime, weight, validator, or evidence is changed.

The warm replay previously established byte-identical `value_17` but different normalized output `x_69`; the first sampled difference was row8, feature23, absolute token40. The cold input at `value_17` differs and must be interpreted separately.

| Output | Stage | Original index | Candidate index | FP32 shape |
| --- | --- | ---: | ---: | --- |
| value_17 | Input | 8450 | 9098 | [1,Q,1280] |
| x_65 | Squares | 8451 | 9099 | [1,Q,1280] |
| var_5669 | Ordered sum | 8888 | 9536 | [1,Q,1] |
| mean_7 | Divide by external norm_divisor | 8889 | 9537 | [1,Q,1] |
| var_5672 | Add epsilon 2^-23 | 8891 | 9539 | [1,Q,1] |
| denom_7 | Square root | 8892 | 9540 | [1,Q,1] |
| var_5675 | Reciprocal | 8895 | 9543 | [1,Q,1] |
| x_67 | Input times reciprocal | 8896 | 9544 | [1,Q,1280] |
| step_47 | Quantization step | 8919 | 9567 | [1,Q,1280] |
| var_5690 | Quotient | 8920 | 9568 | [1,Q,1280] |
| var_5691 | Candidate replacement rounding result | 8921 | 9596 | [1,Q,1280] |
| x_69 | Quantized normalized output | 8922 | 9597 | [1,Q,1280] |

All output descriptors use query axis1, default3, range1..32. The builder checks each producer index/name/type/shape, causal chain, and the specified direct arithmetic inputs. Eleven producer operations are byte-identical to the original baseline. `var_5691` intentionally differs: the original `round` is now the v3 fragment's final `select`. Its candidate producer is pinned to SHA-256 `b6d1baa5896f9f0dcd2b1ed9d95a01481f6326024a967acfff476711267b52ed`, with exact branch/predicate inputs and the original output descriptor preserved. This exception never permits changing that candidate operation.

All46,035 candidate operations remain byte-identical. Only model-description and main-block output lists gain entries; removing them must reconstruct the original candidate protobuf exactly. Endpoints, all8 state interfaces and operations, inputs, weights, and other package files remain unchanged. The builder saves direct input names for each checkpoint so downstream analysis can distinguish equal-input operation differences from propagated differences.

The ordered-sum output brackets a multi-operation reduction. If squares agree and the sum differs, this locates an interval; it does not identify the first unequal addition without inspecting its own inputs. Likewise step construction and replacement rounding contain untapped intermediate operations. Same-policy endpoint and all-state byte controls must pass in every policy/chunk cell before any checkpoint comparison is interpreted. Actual device placement remains unverified.

Root-owned build command:

```sh
.venv/bin/python scripts/build_normalization_taps.py --attempt taps-1 --compile
```

The builder never predicts. The preflight performs only in-memory protobuf mutation and inventory checks, with no native compilation/load/inference or package creation. A structural pass is not a numerical or instrumentation-control pass.
