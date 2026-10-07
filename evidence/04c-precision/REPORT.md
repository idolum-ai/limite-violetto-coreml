# Precision follow-up — live report

The unchanged score limit is 0.04. Original NVIDIA-runtime fidelity is still unavailable; all comparisons here target the qualified local CPU reference.

## Whole-model arithmetic ablation (32-token prefix)

| Linear implementation inside the graph | Maximum final-score error | Gate |
|---|---:|---|
| FP32 then BF16 rounding (previous candidate) | 0.254244 | FAIL |
| FP64 then BF16 rounding | 0.254244 | FAIL |
| FP32 blocks of 32, compensated accumulation | 0.178608 | FAIL |
| Native BF16 linear, remaining graph unchanged | 0 | PASS on this probe |
| Explicit CPU-reference summation order in FP32 | 0 | PASS on this probe |

The native-BF16 control localizes this observed error to linear arithmetic. Restoring generic high precision is insufficient to reproduce a particular floating-point reduction order. The installed Torch build uses Clang 15, which disables the source's BFDOT branch. Its relevant BF16 path uses eight four-lane FP32 accumulators, combines registers in a specified tree, then performs a pairwise lane reduction. The experimental `ordered_linear.py` expresses this order in graph operations. Normal-scale products of BF16 operands fit exactly in FP32; preserving the summation order matters at BF16 rounding boundaries.

The whole graph using that operation matches the reference final scores exactly in Torch on this one recorded prefix. Core ML conversion/runtime checks are now in progress. This is not yet a Core ML gate pass, a proof across all inputs, or an independent original-model fidelity result.

Evidence: `projection-accumulation.json`, `graph-native_bf16.json`, `graph-fp64.json`, `graph-kahan32.json`, `graph-ordered.json`. Pinned source is under `sources/torch-2.7-linear` at installed commit `134179474539648ba7dee1317959529fbd0e7f89`.

## Core ML results and remaining boundary

The initial ordered graph exported but failed runtime shape inference because its dynamically constructed zero accumulator lost a usable output shape. That failure is retained in `attempt1/` and the `0-dynamic-fill.mlpackage`. Initializing from the first real products preserves the symbolic input shape; this second version converts and executes.

The isolated **layer-0 query projection** is exact in Core ML, including its unrounded FP32 result and BF16-rounded output, for sequence lengths 1, 3 and 32 under both permitted compute settings. BF16-rounded normalization and identity rounding also match in these probes. Raw normalization has small differences, demonstrating why unrounded diagnostics and rounded outputs must be distinguished. [Operator evidence](operators.json).

The complete four-layer ordered shard still differs from its Torch graph reference. On CPU, its 32-token prefill relative RMSE is 0.00014264; single-token decode relative RMSE is 0.00489688. GPU-permitted differences are larger. These intermediate magnitudes do not establish the final-score gate. [Shard evidence](shard-runtime.json).

A diagnostic graph exposing individual BF16 rounding sites identifies the first observed divergence:

| Setting | First site | Initial difference |
|---|---|---|
| CPU_ONLY | Layer index 1: attention weighted-value output, before XSA/gating | One element differs by 0.015625 |
| CPU_AND_GPU | Layer index 0: key projection output | Two elements differ; max absolute 0.00390625 |

The CPU path is exact through all earlier observed rounding sites in this diagnostic run. The CPU observation does not yet distinguish attention score multiplication, softmax or the weighted-value reduction as the originating calculation. The GPU-permitted first mismatch suggests that isolated-operator results cannot be assumed to carry over to a composed graph; actual device placement is still unprofiled. Extra diagnostic outputs can alter fusion, so these observations guide diagnosis rather than replacing the original gate suite.

[Complete rounding-site results](rounding-sites.json) · [Source operation labels](rounding-site-labels.json) · [Summary](summary.json)

The selected complete-model packages remain the earlier FP32/BF16-emulated candidate, now protected by the cleanup guard. The ordered candidate is **one experimental shard**, not a silently selected new full model. Full Core ML fidelity still fails; original NVIDIA-runtime fidelity remains unavailable.

## Reproduce and continue

Preserve the existing attempt before rerunning scripts that write fixed evidence paths:

```sh
.venv/bin/python scripts/probe_projection_accumulation.py
.venv/bin/python scripts/ablate_graph_precision.py --mode ordered
.venv/bin/python scripts/convert_coreml.py 0 --arithmetic ordered --output-dir artifacts/04c-precision/ordered --evidence-dir evidence/04c-precision
.venv/bin/python scripts/probe_coreml_shard.py --arithmetic ordered --model-dir artifacts/04c-precision/ordered --evidence-dir evidence/04c-precision
.venv/bin/python scripts/probe_ordered_operators.py
.venv/bin/python scripts/probe_coreml_layer_stages.py
```

The layer-stage probe reuses its existing diagnostic package. Remove it only in a preserved/new attempt when changing graph arithmetic; merely editing the script does not regenerate that package. Its output-name mapping accounts for the compiler aliasing the final output to the last rounding site.

Next: capture identical attention inputs at the CPU mismatch and compare score, softmax and weighted-value operations separately; investigate key-projection execution in the composed GPU-permitted graph. Scale the ordered conversion only after the corrected shard is justified. No threshold widening, no original-runtime fidelity claim, and no performance claim from overlapping diagnostic runs.
