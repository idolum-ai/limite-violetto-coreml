# One-site native group experiment

The isolated division failed C3 saved-baseline reproduction: both policies produce exact reference values there. The separately frozen external-seed C3b gate passed all28 correction calls, including captured wrong seeds. These outcomes remain separate. C3b supports a new in-model diagnostic; it is not a repair claim or a retroactive pass for C3.

Graft the exact exported C3b correction operations after the original `var_5675` reciprocal in the original round-fixed group0 candidate. Rename that original output to `reciprocal_seed`; retain its operation semantics. Route the corrected result to the original output name. Guard the bounded denominator domain and retain the original seed outside it. Preserve all other46,034 operations and all weights/interfaces/states.

Build plain and output-only tapped variants. The latter adds the12 existing normalization taps plus `reciprocal_seed`, with every operation and weight byte-identical to the plain candidate.

Run12 fresh restored-state calls: original plain, candidate plain, candidate tapped × CPU_ONLY/CPU_AND_GPU × saved32-token cold/12-token warm chunks. All arms receive identical saved CPU inputs and before-state.

Mandatory controls:

1. Original plain endpoints and all8 states match saved original candidate results for every policy/chunk.
2. Candidate tapped endpoints and all8 states match candidate plain within every policy/chunk.
3. Candidate CPU endpoints and all8 states match original CPU bytes in both chunks.
4. Interpret taps only if1–3 pass. Pre-reciprocal values and original seed must match the prior controlled same-policy captures.
5. Corrected reciprocal must match exact rational nearest-FP32 reference for each saved denominator; all denominators satisfy the bounded domain and seeds are within one neighbor.
6. Warm normalization stages must match original CPU bytes, including all19 formerly differing final values.

Record group outputs and all cache residuals independently. These can still differ downstream: no full-model or speed claim follows from a local repair. Preserve failed controls and forbid tap interpretation on instrumentation failure. Actual device placement remains unverified.
