# Candidate-2 embedding and shard-0 integration check

The actual candidate-2 `embedding.mlpackage` and `0.mlpackage` accept the expected `norm_divisor` input as shape `[1]` FLOAT32. Shard 0 retains all eight state names (`inner_k_cache_0..3`, `inner_v_cache_0..3`). The [integrated run](results.json) records package Manifest/model/weight hashes, source hashes, and the conversion source-identity hash. Predictions used `CPU_ONLY` and installed the pinned cleanup guard before model load and prediction.

Across three canonical prompts, each split into a 32-token chunk, remainder, and one decode token, embedding `x` and `ve` matched the corrected Torch graph bitwise at all nine steps. Shard-0 `tap23_out` also matched at all nine. Shard-0 `y` matched completely on prompt 0 and on four of six steps for prompts 1–2. The nonexact steps were:

| Prompt and tokens | Different `y` elements | Max abs | Relative RMSE |
| --- | ---: | ---: | ---: |
| Prompt 1, 32:58 | 649 | 256 | 0.0007569816 |
| Prompt 1, decode 58:59 | 155 | 32 | 0.0013879547 |
| Prompt 2, 0:32 | 305 | 64 | 0.0000167803 |
| Prompt 2, 32:48 | 435 | 32 | 0.0001583624 |

The integrated `0.mlpackage` and the earlier independently converted ordered norm/XSA shard package produced **bitwise identical `y` and `tap23_out`** on every one of these nine steps, using independent states and the same actual candidate-2 embedding outputs. See [package comparison](compare.json). These differences therefore reflect input-dependent arithmetic in the graph, not candidate-2 wrapper or interface drift.

## First rounded mismatch on prompt 2, tokens 0:32

The [127-site diagnostic](prompt2-rounding-sites.json) uses the corrected ordered linear, softmax, norm, and XSA graph. Sites 0–80 match the corrected Torch graph bitwise. Site 81 is `Layer.finish` layer 2's `Q(torch.matmul(weights, val))`: one value differs at output `[batch 0, token 31, head 5, channel 20]`, Core ML 2.546875 versus Torch 2.5625. The downstream XSA correction and shard output amplify this difference. This localizes the first **BF16 rounding** difference, but by itself does not establish which FP32 operation upstream of site 81 differs, because softmax weights are not rounded at their own boundary.

The [full-shape product control](full-product/summary.json) fed identical Torch reference weights `[1,10,32,2048]` and cached values `[1,10,2048,128]` to an isolated Core ML weighted-value product. All 40,960 raw FP32 outputs and their BF16-rounded values matched Torch bitwise. The problematic raw value was 2.5546884536743164 and rounded to 2.5625 in both runtimes.

The [full-shape softmax control](full-softmax/summary.json) fed identical Torch reference masked scores `[1,10,32,2048]` to the isolated Core ML ordered softmax. Core ML produced 964 differing FP32 weights (maximum 5.960464477539063e-8). All 32 weights in the problematic head/query row differed. Feeding **these Core ML weights** into Torch's weighted-value product gave raw 2.554687023162842 and BF16-rounded 2.546875 at the problematic element—exactly the wrong rounded result in the combined diagnostic. Feeding Torch weights gave raw 2.5546884536743164 and 2.5625. Thus the softmax weight difference alone is sufficient to cause site 81; the product reduction is not the cause for this case. The isolated controls have different compiler fusion boundaries from the full shard, so this is a causal input test rather than a full compiler-placement claim.

A [single-row/head stage probe](attention/summary.json) showed raw dot, shifted scores, exponential, normalization, and weight differences, but did not reproduce the BF16 output mismatch after shrinking the tensor shape. It verified that removing permanently masked slots leaves the problematic Torch row's scores, weights, raw product, and BF16 output bitwise unchanged. The full-shape controls above retain the candidate's actual 2048-slot capacity.

Scripts: [integration test](../../../scripts/gpu_integrated_shard.py), [package comparison](../../../scripts/gpu_integrated_compare.py), [rounding trace](../../../scripts/gpu_integrated_stages.py), [attention stage probe](../../../scripts/gpu_integrated_attention.py), [full product](../../../scripts/gpu_integrated_full_product.py), and [full softmax](../../../scripts/gpu_integrated_full_softmax.py). Their source hashes are in the JSON evidence. This check covers embedding and shard 0 only; it does not qualify the full 48-layer candidate.
