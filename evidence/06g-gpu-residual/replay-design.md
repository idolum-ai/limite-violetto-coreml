# Layer 1 normalization replay design

Baseline: the separate 06f group0 with signed ties-even rounds and one corrected layer0 reciprocal. Prior expected outputs/states: 06f group-replay-1 candidate arms. Inputs and pre-call state: saved 06d group0-replay-1 cold32/warm12 fixtures, with original CPU state restored before every prediction.

Run exactly eight guarded stateful calls: CPU_ONLY and CPU_AND_GPU, original and output-only tapped clone, cold and warm chunk. Start each call with a fresh state and verify exact restoration. Capture both original endpoints and all eight post-call caches, plus normalization taps on the diagnostic arm. Record package/compiled inventories, source and runtime identities, raw NPZ files and archive hashes. Recheck pinned sources and inventories on completion. Normal process exit and guard count 8 are necessary; numerical gates are separate.

Interpret taps only after all four cells pass both controls: original versus prior same-policy outputs/all states, and tapped versus original endpoints/all states. Preserve all numerical thresholds. The analyst must independently recheck raw archives and controls, compare stages CPU versus GPU policy, and check each reciprocal against exact rational nearest-even FP32 for its own denominator. Equal inputs are required before attributing different outputs to an operation.

A correction is conditional: the site must show an observed reciprocal error with denominator in [0.5,8] and seed within one FP32 neighbor of the exact reference; correct CPU bytes must be retained. If these conditions do not identify a cause, record a negative finding and target the actual earliest divergence. Do not insert a correction solely from graph order. No full-model or hardware-placement claim follows this probe.

Before execution, replay-1-protocol.json pins the worker, builder result, baseline/prior/fixture records, comparator/provenance helpers, design and every input/expected-output archive. Each attempt uses a fresh directory. Historical results remain immutable.
