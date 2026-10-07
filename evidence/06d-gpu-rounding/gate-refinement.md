# Evidence-driven separation of round semantics and native quantizer representation

The original R1 gate (full quantizer bitwise identical to Torch, including signed zeros) remains **failed** for operator-1 and operator-2. It is not retroactively marked passed.

A matched control converted the unmodified `graphs.Q` using the same input suite and conversion settings. Its 40 native arrays are byte-identical to both candidates: the same14/40 Torch-reference failures, entirely -0 to+0, occur in the original quantizer. The zero-path diagnostic preserves its old-endpoint control and shows exact quotient/sign-factor values but zero-sign loss at multiplication. Selecting signed zeros is not established as the cause.

The proposed production change replaces only `round`, leaving the original final multiply intact. Requiring that change to repair the existing outer multiply would expand its scope and could change the qualified native CPU representation. We therefore introduce a separate, prospectively frozen **R1b** gate that tests the actual changed operation and preserves the native baseline:

1. The replacement round output must match Torch round byte-for-byte, including halfway ties, adjacent values, literal signed zero and negative finite values rounding to zero, under both policies.
2. The full quantizer using that replacement must match the unmodified native CPU quantizer byte-for-byte on the same20 inputs. Continue reporting its strict full-quantizer Torch comparison independently; do not hide canonicalized zeros.
3. Require the original native CPU round control to match the Torch round reference for these cases, or stop and investigate further.
4. A separate uninstrumented group candidate must preserve original native CPU outputs and all eight states byte-for-byte. GPU residuals are measured against that reference. All full-model score/hidden/token criteria remain unchanged.

This is a new, more specific diagnostic gate supported by the matched baseline control; it is not a claim that full Core ML quantization preserves every Torch zero bit. No qualified runtime, source model or acceptance threshold is changed. R1b will be frozen before execution and its source/protocol snapshots retained.

Evidence: [original native quantizer](operator-baseline-1/results.json), [candidate1](operator-1/results.json), [candidate2](operator-2/results.json), [controlled zero path](zero-path-2/results.json).
