# Static inventory of the qualified candidate

This inventory is bound to candidate identity `5a3e4c7a…ad2c6` and was generated from the sealed `.mlpackage` specs and build record. The script verifies the package manifest, then reads MIL specs with `coremltools.models.utils.load_spec`; it does not create `MLModel`/`CompiledMLModel` objects, compile, or run predictions. Reproduce it with:

```sh
.venv/bin/python evidence/05d-decode/static-inventory/analyze_specs.py
```

The source package set is about 5.05 GB on disk, with about 4.98 GB of external weight files. The embedding package is about 889 MiB and the head about 741 MiB; each four-layer group is about 265 MiB. These are package file sizes, not measured resident memory or compiled-cache sizes.

All 12 group conversion commands carry the qualified graph settings recorded in `summary.json`: ordered linear, softmax, norm, and XSA arithmetic; lookup GELU; source exponential; dynamic-product queries; paired value product; and native tiny-score policy. Each group has 42,498–44,476 MIL operations, including 68 `matmul`, 16 `read_state`, 8 `write_state`, and 8 `slice_update` ops. The repeated graph size is consistent with four transformer layers per package; it does not indicate how Core ML schedules or fuses those operators.

The group cache declares eight FP16 states with shape `[1, 2048, 2, 128]`, or 8 MiB of key/value state per group and 96 MiB across 12 groups for one request, before runtime overhead or implementation copies. The embedding and head have no recurrent state.

## Decode-shape hypothesis

In [dynamic_queries.py](../../../coreml_port/dynamic_queries.py), `prepare` chooses `min(32, last_absolute_position + 1)` rows and pads to that size. Thus a one-token decode at absolute position 32 or later reaches attention with 32 query rows. [paired_product.py](../../../coreml_port/paired_product.py) pads to 32 rows, computes 16 two-row matrix products, then trims the output to the requested query count. [graphs.py](../../../coreml_port/graphs.py) applies those query and value paths in each of the four layers in a group.

Using the graph dimensions (10 expanded attention heads, 2,048 key positions, width 128), this implies about 83.9 million multiply products for QK and another 83.9 million for the attention-value product per layer at Q=32. Across 12 groups and four layers per group, the two attention products total about 8.05 billion multiply products per decoded token. A 32-token prefill has the same per-group call shape but yields 32 real tokens, about 251.7 million attention multiply products per generated token when amortized over that prefill.

These figures are shape-based arithmetic counts, not a runtime cost model. They exclude linear projections, MUDD, normalization, softmax, masking, data movement, call overhead, and any backend optimizations. They identify a testable decode-shape hypothesis only; the paired control/profile component timings must determine the observed hotspot.
