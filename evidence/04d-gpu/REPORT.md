# GPU-permitted key projection diagnosis

This probe uses the recorded 32-token local input and the layer-0 scaled BF16-valued weights. It does not change the active runtime, acceptance thresholds, or selected full-model packages. Core ML predictions installed `coreml_port.cleanup_guard.install()` first. All models use FP32 compute precision and macOS 15 minimum deployment target.

## Finding

The layer-0 key mismatch is an exact BF16 midpoint decision in a GPU-planned composed graph. The isolated key, qk, and kv graphs are exact under CPU_ONLY and CPU_AND_GPU, and Core ML plans their nonconstant operations for CPU. The qkv graph plans its nonconstant operations for GPU under CPU_AND_GPU. Its key raw output is exact, but two rounded elements at `[0,12,199]` and `[0,14,199]` become `0.79296875` instead of Torch's `0.7890625`. Both exposed raw values are exactly `0.791015625` (`0x3f4a8000`), the midpoint. Torch/CPU select the even lower BF16 neighbor. The GPU-planned qkv path selects the upper neighbor. The other q and v outputs are exact. [Composition results](key-projection.json), [qk/kv controls](key-projection-qk-kv.json), [midpoint values](key-ties.json), [compute plans](compute-plan.json).

A standalone round-tie model is exact under both compute settings, but its compute plan selects CPU. It therefore does not test GPU midpoint semantics. Exposing the qkv raw output can also change compilation. These observations establish the composed, GPU-planned failure and the exact midpoint; they do not distinguish a GPU round kernel's tie rule from compiler fusion or a separate internal key intermediate.

## Local correction candidate

[`scripts/gpu_even_round.py`](../../scripts/gpu_even_round.py) expresses BF16 round-to-even with `floor`, the fractional part, and parity at an exact half. In the GPU-planned qkv graph, replacing only key rounding makes raw/q/k/v exact on 1, 3, and 32 tokens under both compute settings. The original qkv graph differs only at the two rounded key elements for the 32-token GPU-permitted case. [Multi-length results](key-validation.json), [explicit-even model](key-even.json). The first expression used a Torch boolean `__or__` that Core ML Tools 9.0 could not convert; that failure is [retained](key-even-conversion-failure.json). The working expression uses `torch.where`.

The uninstrumented four-layer ordered shard with only layer-0 key rounding changed matches the original ordered Torch shard exactly for prefill 0:32 and decodes 32:33 and 33:36. Core ML still differs from the original Torch output:

| Compute setting | Span | Original relative RMSE | Corrected relative RMSE |
|---|---|---:|---:|
| CPU_ONLY | 0:32 | 0.000142641 | 0.000142641 |
| CPU_ONLY | 32:33 | 0.004896878 | 0.004896878 |
| CPU_ONLY | 33:36 | 0.004631497 | 0.004631497 |
| CPU_AND_GPU | 0:32 | 0.001394988 | 0.001394181 |
| CPU_AND_GPU | 32:33 | 0.008447301 | 0.008410629 |
| CPU_AND_GPU | 33:36 | 0.008383369 | 0.008400863 |

The key-only correction removes the identified local mismatch but does not pass the shard fidelity boundary. [Corrected shard evidence](shard0-even.json), [baseline shard evidence](../04c-precision/shard-runtime.json). A diagnostic exposing the key raw and rounded site finds zero differences for each under CPU_ONLY and CPU_AND_GPU; its final shard output still differs. Extra outputs can alter fusion, so the diagnostic supports site localization but is not a replacement for the uninstrumented gate. [Key-site diagnostic](keytap.json).

The corrected uninstrumented shard compute plan anticipates GPU for 7,483 nonconstant operations, including 126 `round` operations; the key-site diagnostic anticipates GPU for 7,446. The remaining operations report unknown placement, mostly constants. This is an anticipated plan, not a runtime execution trace. [Shard plans](shard-compute-plan.json).

## Checkpoints and scope

The uninstrumented shard was built from `coreml_port/graphs.py` SHA-256 `8149e6e6132b0b816405735b740cf3bcab0e43734d963905129efc86628462ce`, `coreml_port/ordered_linear.py` `699e1c9c5e7f57879ad009cd4a10498162dbab25f599aa46b21eef62ea2106f2`, and the then-current round module recorded in [shard0-even.json](shard0-even.json). The later key-site diagnostic used `graphs.py` SHA-256 `9ae902e9794e8311b60af091ea59027f8169cc9750f9d48ff914e65226f5f414`; that revision added a default `softmax` injection point whose default is the original `x.softmax(-1)`. Its exact source/module/input hashes are in [keytap.json](keytap.json). Packages live in `artifacts/04d-gpu/`; the shard report contains per-file package hashes. No full-model fidelity or performance claim follows from these probes.

The next GPU correction trial should combine explicit key round-to-even with the separately diagnosed attention correction, then check the earliest rounding site and uninstrumented shard again. Only a passing shard justifies scaling conversion and rerunning final gates.
