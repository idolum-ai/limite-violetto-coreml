# Reduced four-tap residual localization

This fresh output-only builder instruments the same round-fixed group0 candidate as the earlier seven-tap attempt. That attempt and all frozen scripts, artifacts, and evidence remain intact. Its GPU load rejected data-dependent outputs `scores_1`, `weights_1`, and `x_37`; this reduced attempt omits those three outputs.

The pinned source remains `artifacts/06d-gpu-rounding/group0-1/0.mlpackage`, with completed build record SHA-256 `4e6ac1baa2f70c10bf812523f7e341d9dabf34b30299e4819631f2db31cf150f` and exact package inventory binding.

| Retained output | Stage | Original index | Candidate index | FP32 default shape |
| --- | --- | ---: | ---: | --- |
| q_3 | Layer0 final rotary query | 1679 | 1922 | [1,3,10,128] |
| value_19 | Layer0 final hidden | 10698 | 11589 | [1,3,1280] |
| x_93 | Layer1 normalized projection input | 11170 | 12088 | [1,3,1280] |
| x_97 | Layer1 raw key projection before Q | 11776 | 12721 | [1,3,256] |

Each retained value has its dynamic query dimension on axis1, default3, range1..32. The builder checks the exact MIL shape, producer name/index/type, original producer bytes, and causal dependencies. All46,035 candidate operations must remain byte-identical. Only model-description and main-block output lists may gain entries; removing them must reconstruct the entire original candidate protobuf exactly. Original endpoints, inputs, all8 states, weights, and every package file except the model protobuf stay unchanged.

The root runner can build a fresh attempt with:

```sh
.venv/bin/python scripts/build_rounding_residual_taps_v2.py --attempt residual-taps-2 --compile
```

The builder's success establishes structural preservation only. Both compute policies must still load and predict successfully, and their instrumented original endpoints plus all8 post-states must match the untapped candidate byte-for-byte on each chunk before interpreting tap differences. Omitting the rejected outputs does not by itself establish GPU load compatibility. Native compilation and prediction are root-owned; no native job is part of this builder preflight.
