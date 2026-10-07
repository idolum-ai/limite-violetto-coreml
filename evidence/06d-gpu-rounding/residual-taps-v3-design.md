# Six-tap layer0 residual and MLP localization

The four-tap residual replay passed both-policy endpoint/state controls and found exact layer0 rotary queries but different layer0 final hidden values. This fresh output-only attempt narrows that interval. All previous builders, artifacts, and evidence remain unchanged.

The source is still `artifacts/06d-gpu-rounding/group0-1/0.mlpackage`, pinned by completed build record SHA-256 `4e6ac1baa2f70c10bf812523f7e341d9dabf34b30299e4819631f2db31cf150f` and its complete package inventory.

| Output | Stage | Original index | Candidate index | FP32 default shape |
| --- | --- | ---: | ---: | --- |
| q_3 | Layer0 final rotary query | 1679 | 1922 | [1,3,10,128] |
| value_17 | Attention residual after Q | 8450 | 9098 | [1,3,1280] |
| x_69 | MLP input normalization | 8922 | 9597 | [1,3,1280] |
| x_79 | Gated MLP activation after Q | 9606 | 10389 | [1,3,3328] |
| mlp_1 | Down projection after Q | 10617 | 11427 | [1,3,1280] |
| value_19 | Final layer0 hidden | 10698 | 11589 | [1,3,1280] |

All six have query dimension axis1, default3, range1..32. The dependency checks follow q_3→value_17→x_69→x_79→mlp_1→value_19 and additionally verify the residual dependency value_17→value_19. The attention residual incorporates the original input tensor; subsequent operations preserve the query axis while changing or reducing features. No raw scores, softmax weights, data-dependent attention-slice outputs, x_55, or out_1 are exposed.

Every tap's producer name, original/candidate index, type, shape, and original producer bytes are checked. All46,035 candidate operations remain byte-identical. Only the model-description and main-block output lists may change; removing added entries reconstructs the complete original candidate protobuf exactly. Existing outputs, all8 states, input shapes, weights, and other package files remain unchanged.

Root launch command:

```sh
.venv/bin/python scripts/build_rounding_residual_taps_v3.py --attempt residual-taps-3 --compile
```

This structural check does not establish instrumentation validity. Each policy must load and predict, and instrumented `y`, `tap23_out`, and all8 post-states must match the untapped candidate byte-for-byte on both chunks before tap comparisons can be interpreted. The builder never predicts; no native job is part of its preflight.
