# Layer-2 CPU normalization follow-up

This bounded probe uses the ordered linear and ordered softmax graph and the canonical 32-token input. It targets the first CPU_ONLY mismatch in `evidence/04d-cpu/ordered-softmax-stages`: round site 69, the normalization entering layer 2. It does not modify shared runtime code or the separately built full candidate.

The tensor leaving layer 1 at round site 68 matches Core ML exactly. Feeding that identical tensor into an isolated normalization graph shows exact elementwise squares. The first arithmetic difference is the 1,280-element mean: 17 of 32 rows differ, maximum absolute difference 128 on a large row. The following `rsqrt` differs at 22 rows, and BF16 rounding differs at ten elements, all on row 9. [Stage results](norm-stages.json), [captured input](../../../artifacts/04d-gpu/cpu-norm/norm-input.npz).

Torch's `sum / 1280` reproduces its `mean` on all 32 captured rows; `sum * (1 / 1280)` differs on five rows. Core ML's raw sum itself differs on 19 rows, and passing a runtime denominator to `real_div` still leaves 16 mean differences. The runtime divisor fixes row 9's mean, but rounded normalization still differs because Core ML `rsqrt` is one FP32 step below Torch at that row. [Reduction-order screen](reduction-orders.json), [sum/division split](sumdiv.json), [dynamic denominator only](dynamic-norm.json).

With the exact Torch mean supplied directly, Core ML `rsqrt` differs on 18 rows and yields the ten rounded mismatches. `ones_like(mean) / sqrt(mean)` matches the Torch `rsqrt` output exactly on all 32 rows and yields exact BF16-rounded normalization. A `pow(-0.5)` form has raw differences but also rounds exactly. [Reciprocal-square-root alternatives](rsqrt.json). Replacing `rsqrt` alone while retaining ordinary Core ML mean does not clear the rounded mismatch. [Simple sqrt/div probe](sqrt-only.json).

Combining a runtime denominator with `ones_like(mean)/sqrt(mean+EPS)` clears all BF16-rounded differences on the isolated layer-2 input under CPU_ONLY and CPU_AND_GPU. Mean and raw normalized values still differ on other rows, so this is a tested boundary correction, not general arithmetic identity. [Combined isolated probe](combined.json).

An experimental four-layer shard changes only the ninth normalization call, which is layer-2 entrance, and adds a runtime divisor input containing 1280. Its Torch output remains exactly equal to the original ordered Torch shard for prefill 0:32 and decodes 32:33 and 33:36. The uninstrumented Core ML CPU_ONLY relative RMSE improves:

| Span | Ordered-softmax baseline | Combined norm candidate |
|---|---:|---:|
| 0:32 | 0.000123371 | 0.000011100 |
| 32:33 | 0.002462602 | 0.002415277 |
| 33:36 | 0.002616964 | 0.002120592 |

The corrected shard still differs from Torch. GPU-permitted relative RMSE is 0.001395020, 0.008523712, and 0.008401641 for those spans. [Shard evidence and package hashes](shard0.json), [baseline shard](../../04d-cpu/ordered-softmax-shard/shard-runtime.json). The shard package is `artifacts/04d-gpu/cpu-norm/shard0.mlpackage` and uses a changed input signature solely for this experiment. The active full candidate remains separate.

The shard source hashes are recorded in `shard0.json`: `graphs.py` `9ae902e9794e8311b60af091ea59027f8169cc9750f9d48ff914e65226f5f414`, `ordered_linear.py` `699e1c9c5e7f57879ad009cd4a10498162dbab25f599aa46b21eef62ea2106f2`, and `ordered_attention.py` `3abc03e63175f431ff00d17f68eb0c976c1b92417ab29fa411c1b57317ae8b44`. The reusable expression and bounded conversion harness are in [`gpu_cpu_norm_shard.py`](../../../scripts/gpu_cpu_norm_shard.py). Further source-order mean work belongs to the separate CPU investigation; the full gate, rather than this local probe, decides candidate readiness.
