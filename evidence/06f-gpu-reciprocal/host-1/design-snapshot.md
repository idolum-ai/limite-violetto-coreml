# Boundary-safe reciprocal candidate: host-first design

This new candidate corrects a supplied FP32 seed instead of rounding a Newton correction sum. The API is `correct_reciprocal(d,r0)`; diagnostic intermediates and a matching MIL Builder function are available separately. It never replaces or computes the seed, allowing separate native tests of captured CPU/GPU-permitted seeds and of a live original division. Qualified arithmetic, runtime, prior failed candidates, and evidence remain unchanged.

Contract: d is positive finite FP32 in [0.5,8]; r0 is a positive normal FP32 value at most one representable neighbor away from nearest-even FP32(1/d). Input shapes match. This is not a general division or reciprocal function. Every host test checks this contract using an exact rational oracle. No FP64 arithmetic, casts, bit reinterpretation, division, or logarithms are used inside the correction.

1. Split d and r0 using4097 and obtain an error-free product pair p+err=d*r0 with the standard Dekker product-error sequence. Here p is near1, so h=1-p is exact by Sterbenz.
2. Construct the immediate-neighbor spacings using comparisons with powers of two and exact constants. Downward spacing is half upward spacing at a power boundary. Include seed neighbors immediately below1/8 and above2.
3. The upper reciprocal midpoint is r0+up/2; compare h to err+d*up/2. The lower midpoint is r0-down/2; compare h to err-d*down/2. Multiplication by either half-spacing is exact power-of-two scaling.
4. Represent each threshold with TwoSum as hi+lo. Compare h with hi; if equal, use lo's sign. Never sum hi+lo again, preserving the small component that the previous final compensated correction lost.
5. On an exact midpoint tie, choose the even-significand endpoint. Parity uses exact power-of-two scaling followed by floor. Return the already representable previous seed, original seed, or following seed; no final arithmetic correction sum is rounded.

The correction uses only FP32 add/subtract/multiply/floor, fill_like, comparisons, boolean operations, and select, all available to the installed MIL Builder. The host proof harness checks each exact product, exact residual, exact midpoint scaling, and exact TwoSum decomposition against rational arithmetic; it also checks immediate neighbors and parity against FP32 bits. Native compiler contraction/reassociation or arithmetic behavior remains a separate qualification risk. A host pass proves only the declared finite test set and verified arithmetic identities, not native device placement or full-model fidelity.

The test cohort is predeclared before execution:

- Preserve all108 denominator inputs from the previous four-case host attempt, including its three failed power-predecessor cases.
- Add the ±256-FP32-neighbor windows around d=0.5,1,2,4,8, clipped to the declared domain:2053 boundary values.
- Add8192 fixed-seed (`20260924`) random positive FP32 bit patterns drawn uniformly from the inclusive bit interval for [0.5,8].
- Exercise every denominator using correctly rounded reciprocal, its predecessor, and its successor as supplied seeds.
- Separately exercise all88 saved native seeds: warm CPU/GPU (12 each), cold CPU/GPU (32 each), preserving their captured policy provenance.

Thus the frozen host plan has10,353 denominator entries,31,059 oracle/neighbor-seed corrections, plus88 captured-seed corrections:31,147 corrections total. Duplicate inputs are intentionally preserved. No adaptive random sampling, exclusions, numerical tolerance, or weakened CPU-preservation gate is permitted. Host results and exact input hashes will be saved before any native model conversion. This task performs no native build or inference.
