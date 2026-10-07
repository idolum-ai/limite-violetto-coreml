# CPU precision milestone — live report

The authorized milestone is a complete CPU Core ML runner passing the existing local-reference gates. The score maximum-error limit remains 0.04. This phase does not establish original NVIDIA-runtime fidelity.

**Current status: CPU fidelity milestone PASS.** Candidate 6 passes all frozen local-reference gates on the pinned macOS 26.5.2 arm64 host: 54/54 short comparisons, 97/97 partition checkpoints, and 6/6 window/capacity comparisons through 2,048 tokens. All 142 hidden-state comparisons are exact. Short, partition and boundary lifecycle/disposal/repeat checks pass.

Generation completed with the correct boxed answer 391 after 674 tokens, all identical to the qualified CPU reference. Seven full-prefix score audits pass; maximum score difference is 3.814697265625e-6, below the unchanged 0.04 limit. The explicit tokenizer EOS stop policy is retained. The aggregate checker verifies the sealed package/source identity and matching runtime across all gates, with no failed or missing gates.

| Gate | Result | Evidence |
|---|---|---|
| Source/package provenance | PASS, 14 packages | [Seal](full-candidate-6/provenance-verification.json) |
| Short numerical and lifecycle | PASS, 54 comparisons and 17 checks | [Summary](full-candidate-6/summary.json) |
| Request partitioning | PASS, 97 comparisons, 13 disposals, exact repeat | [Summary](../04d-audit/candidate6-partitions/summary.json) |
| Window and capacity | PASS, 6 comparisons and 4 checks | [Summary](full-candidate-6-boundaries/summary.json) |
| Completed generation | PASS, 674 matching tokens and 7 audits | [Summary](../04d-audit/candidate6-generation/summary.json) |
| Aggregate milestone | PASS, no missing or failed gates | [Machine-readable result](candidate6-milestone.json) |

Candidate identity: `5a3e4c7a571e4fc3532377d9b9bb0da023f499c10709216fee7e1001bf4ad2c6`. The corrected score reduction is host/backend specific. This is bounded qualification against the local native CPU reference, not original NVIDIA-runtime fidelity or broad task accuracy. GPU and Neural Engine fidelity, clean-machine installation, and performance usefulness remain unqualified. Both pinned process-local runtime guards are required. Default package selection remains historical; [README](../../README.md) provides explicit candidate selection.

The generation audit took 1,857.76 seconds including load and reference checks. This is a correctness-run duration, not a controlled performance benchmark. Step 5 should measure startup, prefill, decode, memory, and task usefulness before device optimization or packaging.

Historical experiments and failed candidates are retained below.

## Initial attention evidence

- Captured identical real layer-1 attention inputs from the ordered Torch graph for the canonical first prompt's 32-token prefix. Inputs: `../../artifacts/04d-cpu/attention-inputs.npz`.
- CPU Core ML attention dot products and scaled scores match exactly. With identical reference softmax weights, the value product and BF16-rounded output also match exactly.
- Softmax is the first differing operation in this isolated test. Its maximum weight difference is about 2.4e-7. A combined diagnostic graph crosses two BF16 rounding boundaries; the earlier composed shard diagnostic crossed one. Diagnostic outputs can alter compilation, so neither replaces uninstrumented runner validation.
- `baseline-1` retains valid JSON metrics but its per-operation NPZ evidence names collided. Corrected the harness and retained a separate complete `baseline-2` run; no old evidence deleted.
- Downloaded CPU softmax source at installed PyTorch commit `134179474539648ba7dee1317959529fbd0e7f89`. The next experiment expresses its exponential, four-lane sum order, reciprocal and multiplication separately.

## Parallel work and validation boundary

Sol initially investigated GPU projection rounding and now investigates the CPU tiny-prefix score path in `../04d-gpu/`. The independent audit in `../04d-audit/REPORT.md` identifies package selection/identity and missing matched long-context coverage as critical gates. The old complete model remains selected; no partial candidate is promoted.

## Evidence links

- [Baseline summary](baseline-2/summary.json)
- [Decomposed softmax experiment](decomposed-1/summary.json)
- [Audit](../04d-audit/REPORT.md)

Uninstrumented shard validation must precede complete-model conversion, fixed prompt and teacher-forced score gates, cache/window boundaries, lifecycle checks and generation. Passing isolated operations is insufficient.

## Source-order softmax result

The decomposed experiment exposed an additional converter behavior: Torch `1/total` becomes MIL `inverse`, whose default epsilon is 1e-4. Explicit `ones_like(total)/total` uses real division and avoids that change. The failed epsilon-bearing experiment is retained in `decomposed-1`; corrected evidence is `decomposed-2`. Masked diagnostic logits compare negative infinity with a finite negative saturation value; the JSON consequently contains nonfinite diagnostic statistics for that output only. Both exponentiate masked entries to zero. Exponential values still differ slightly, but all rounded attention outputs match on this isolated capture.

`ordered_attention.py` mirrors the pinned CPU four-lane sum order. With ordered linear and this softmax, the complete 48-layer Torch graph matches all recorded intermediates and final scores exactly on the original 32-token probe. [Torch graph](ordered-softmax-shard/graph-ordered.json).

The new uninstrumented CPU Core ML shard still differs, with relative RMSE 0.00012337 for prefill, 0.00246260 for one-token decode, and 0.00261696 for three-token decode. These are within the existing hidden-state limit for this probe, but are not final-score acceptance. Decode error is roughly half the previous ordered candidate. [Shard evidence](ordered-softmax-shard/shard-runtime.json).

The composed diagnostic now matches the first 69 BF16 rounding sites. First difference moves to layer 2 input normalization: 10 elements, maximum 0.015625. Sol is investigating that boundary. [Stage evidence](ordered-softmax-stages/rounding-sites.json).

## Full trial and package selection

A separate complete candidate is being built in `artifacts/04d-cpu/full-candidate-1`, using unchanged embedding, the tested shard 0, and newly converted remaining groups/head. This is an experiment justified by improved shard errors, not a candidate promotion. [Build/source records](full-candidate-1/build.json).

The runtime now accepts an explicit candidate root only with a package-identity manifest. It verifies all 14 packages before loading; the validator records the actual root and identity. Default package selection remains historical. [Identity helper checks](../04d-audit/REPORT.md), [runtime rejection checks](../04d-audit/runtime-selection-checks.json).

All 14 packages converted. The complete trial's aggregate identity is `b99594500b475e5423db85a2efaceba57c6c7d15738cef476b14de4f69a017c1`. Post-build source hashes match the pre-build records; copied embedding/shard packages are byte-identical to their recorded sources. [Provenance check](full-candidate-1/provenance-verification.json).

Full CPU validation completed: **30/54 numerical comparisons pass; all 15 score comparisons fail; all 15 tested top-1 tokens match; all 17 lifecycle checks pass**. Worst score error is **0.3500743** against 0.04, versus the historical candidate's 0.3350806. The local shard improvement did not improve this complete-model worst-case result. Candidate 1 is **not promoted**. No longer-context or generation success is inferred from this failed short gate. [Complete validation](full-candidate-1/validation-CPU_ONLY.json), [summary](full-candidate-1/summary.json).

## Further normalization diagnosis

The exact layer-2 input and its elementwise squares match. Both the mean reduction and reciprocal-square-root implementation differ on CPU Core ML. Given exact reference means, explicit `ones_like(d)/sqrt(d)` fixes the isolated reciprocal/root rounding boundary. Keeping the ordinary mean is insufficient: its row-9 error still yields ten BF16 mismatches. A combined runtime-divisor sum/divide plus sqrt/divide probe clears all rounded mismatches on this capture; raw values still differ on other rows. Sol is testing a bounded composed shard correction. These later experiments do not change full candidate 1. [CPU normalization evidence](../04d-gpu/cpu-norm/combined.json).

Root then implemented the pinned CPU `SumKernel.cpp` order for the two normalization widths (128, 1280). The captured Core ML sum, mean, reciprocal root, raw normalization and rounded output now all match exactly. Four Torch source checks (captured, independent widths, scaled inputs) match exactly. A separate **uninstrumented** operator suite covers both widths, 1/3/32 token counts and three scales on CPU and GPU-permitted settings. [Source-order probe](ordered-norm/summary.json), [shape suite](ordered-norm-shapes/summary.json). These are operator tests, not full-model qualification.

The audit branch is also checking the pure Torch graph on actual multi-chunk prompts and cached decode, to distinguish graph/cache differences from compiler differences before scaling another candidate.

## Exact corrected shard and second complete trial

The pure Torch audit is now exact on all three canonical prompts (44/58/48 tokens), persistent full-capacity FP16 caches, and four matched decode steps each: all 15 score comparisons and selected hidden comparisons have zero error. This holds for both the first candidate arithmetic and the later complete set of corrections. [Fully corrected Torch audit](../04d-audit/torch-candidate/candidate2-ordered-all-canonical3/results.json). It excludes the tested graph/cache path as the source of these short-case differences; long-context parity remains separate.

Source-ordered RMS normalization fixes both tested shard decode spans exactly, leaving a small prefill difference. That prefill difference localizes to the XSA correction. Reproducing the pinned CPU norm2 four-lane reduction and dot-product reduction makes the **ordinary four-layer Core ML CPU shard exact on 32-token prefill, one-token decode, and three-token decode**. [XSA diagnosis and exact shard](../04d-gpu/xsa/REPORT.md). GPU-permitted execution still differs; it is not qualified by this CPU result.

A bounded exhaustive activation screen covers 10,243 BF16 inputs. Core ML rounded sigmoid and SiLU match the float graph in that screen; GELU differs on 174 values, maximum 1.19e-7. The float graph itself differs from native BF16 sigmoid/SiLU on 38/30 grid values, so this is not universal BF16 emulation. Those differences are retained in the evidence. [Activation screen](activations/summary.json).

A bounded GELU lookup matches the float graph at all 10,247 grid/tail controls on CPU and GPU-permitted settings. [Lookup validation](gelu-lookup/summary.json). Combined with the source-ordered linear, softmax, RMS and XSA operations, it also preserves all 15 full-model Torch score checkpoints exactly.

Full candidate 2 is now building in its own folder, with an explicit `norm_divisor` scalar input for each embedding/group package to preserve real division. The runtime supplies 1280 only when that input is declared; 128-wide norms derive their divisor within the graph. The ordered head is copied from candidate 1 and will be digest-checked. [Build](full-candidate-2/build.json). Default package selection is unchanged.

## Candidate 2 outcome and source exponential

Candidate 2 converted all 14 packages and passed provenance checks, aggregate identity `9221318ea33c0223c4e7fc491b5483b2eb6b3cc4e099af700fa18f9712b4cb81`. The complete CPU trial **fails**: 36/54 numerical comparisons pass, 0/15 score comparisons pass, maximum score error 0.4056477547 against 0.04, all 15 top-1 choices match, and 17/17 lifecycle checks pass. It is not promoted. [Summary](full-candidate-2/summary.json), [validation](full-candidate-2/validation-CPU_ONLY.json).

The integrated shard is bitwise identical to the independently converted shard. Prompt 2 reveals a first differing BF16 site in layer 2 attention. Identical reference weights make the full-shape Core ML product exact. Conversely, feeding Core ML weights to Torch reproduces the wrong rounding. This establishes a causal softmax-weight difference for this site. [Controlled diagnosis](../04d-gpu/integrated-shard/REPORT.md).

Re-expressing the pinned SLEEF exponential polynomial with compensated FP32 arithmetic matches Torch exponential exactly on 30,245 captured/independent inputs. Naive multiply/add does not; the failed attempt is retained. This compensation is qualified empirically for the bounded nonpositive exponential domain, not as a general correctly-rounded FMA implementation. Source and Boost license are retained under `sources/sleef-pinned/`. [Initial probe](sleef-exp-2/summary.json).

A separate converted graph, traced with zeros and then fed fresh values, matches exactly on 540,672 values spanning uniform [-104,0], log-uniform negative magnitudes, range-reduction boundaries and masking tails. [Independent dynamic-input suite](source-exp-independent-1/summary.json). The corrected full-shape softmax matches all reference weights and the formerly failing rounded attention result exactly. [Causal correction](../04d-gpu/source-exp/full-softmax/summary.json). Composed shard and full-runner validation remain required.

The source-exponential shard passes all nine tested steps across the three canonical prompts bitwise, for both ordinary outputs. The full pure-Torch graph also remains exact at all 15 canonical prompt/decode score checkpoints. [Shard evidence](../04d-gpu/source-exp/shard0/summary.json), [Torch audit](../04d-audit/torch-candidate/candidate2-sourceexp-canonical3/results.json).

Candidate 3 is now building in an isolated root, copying candidate 2's embedding/head and converting all 12 groups with the source exponential. Math/converter sources are frozen during the build; the sealing check will verify them afterward. [Build record](full-candidate-3/build.json). No promotion has occurred.

A separate direct-prefix pure-Torch preflight matches scores exactly at lengths 1, 3, 15, 16, 31, 32 and 33. [Results](source-exp-tiny-torch-2/results.json). The first harness failed by referencing a nonexistent convenience attribute before saving any comparisons; that log remains in `source-exp-tiny-torch.log`, and the fixed attempt uses the established 151667 vocabulary bound. This is graph evidence only; actual Core ML partition gates are still pending.

Candidate 3 converted all 14 packages, aggregate identity `93ba0bc3e64ccc4b04f0faa5c3e1953d062743ffe4c862af872c1030b95d0966`. [Post-build provenance verification](full-candidate-3/provenance-verification.json) passes; conversion sources stayed unchanged, and copied embedding/head packages match their recorded sources. Complete CPU validation is running.

## Candidate 3 passes the complete short gate

**54/54 numerical comparisons and 17/17 lifecycle checks pass.** All 39 captured hidden-state comparisons are bitwise exact. All 15 score comparisons pass, maximum error 1.9073486328125e-06 against 0.04, with all top-1 choices matching. [Summary](full-candidate-3/summary.json), [complete validation](full-candidate-3/validation-CPU_ONLY.json). This qualifies the short suite only. Root is running the matched window/capacity gate and the independent audit is running tiny/irregular partition gates in parallel. Their timings are not performance measurements.

## Extended gates: real failures and an audit-harness correction

The native-BF16 full-prefix window gate passes lengths 1023, 1024 and 1025 with exact final hidden states and score errors at most 0.00000190735. Length 1026 fails with score error 0.1835803986. The test continues through full capacity and a fresh partition. [Live boundary results](full-candidate-3-boundaries/results.json).

The independent tiny fixture `[87555, 130664, 114854]` also exposes a real graph discrepancy. A separate pure-Torch graph reproduces it, first at layer 32; changing the three-key softmax sum order does not change the result, and FP32-cache ablation also leaves it unchanged. [Tiny diagnosis](../04d-gpu/tiny3-scalar-softmax/summary.json), [first layer](../04d-gpu/tiny3-first-layer/summary.json), [cache ablation](../04d-gpu/tiny3-cache-ablation/summary.json).

Root review found two errors in the first partition audit harness: it cast reference hidden states to FP32 before the head, changing reference head arithmetic; and it reused one longest-prefix reference for shorter API endpoints despite known CPU shape dependence at tiny lengths. The old attempt is retained as invalid gate evidence; its approximate 0.029 score residuals are not model defects. The independent native-BF16 tiny3 graph discrepancy remains real (score error about 0.38456), and the separate long-boundary validator does not have those harness errors. A corrected partition audit must compute each full-prefix endpoint and preserve BF16 through the reference head.

The long gate completed: 1023/1024/1025 pass; 1026 and fresh1026 both fail at 0.1835803986; capacity 2048 fails at 0.5038032532. All four lifecycle/partition-repeat checks pass. The complete pure-Torch graph at fresh1026 is exact against native BF16 reference at every captured last-token group and score, isolating the fresh long-case discrepancy to conversion/runtime. [Torch preflight](source-exp-long-torch/results.json).

The corrected partition audit confirms direct tiny3 failure at score error 0.384563. Other direct lengths 1/15/16/31/32/33 pass with exact hidden and score errors at most 0.000003815. Cached prefixes 3–14 fail on the affected fixtures; later checkpoints align. The [native-reference audit](../04d-audit/candidate3-partitions-native-ref/results.json) supersedes earlier invalid harness attempts. Fresh33 with partitions [33] and [1,32] is bitwise identical.

A new opt-in runtime adapter mirrors the qualified CPU's bounded prefix replay below 16 tokens, with six focused lifecycle checks passing. It does not fix direct tiny3 by itself and is not selected by default. [Adapter report](../04d-audit/REPORT.md).

For direct tiny3, an isolated dynamic-query Core ML layer32 with raw-layout attention scores and fixed full-capacity keys now matches both attention and layer output exactly. An ordinary four-layer composed test is required next; the Torch expression itself still differs, so the evidence specifically concerns compiled CPU arithmetic. [Layer gate](../04d-gpu/tiny3-dynamic-layer32/summary.json).

## Extended failure localized and bounded query-shape corrections

The uninstrumented dynamic-query four-layer group8 now fixes all 789 differing hidden values on the frozen direct tiny3 fixture. Its original row policy uses min(32, absolute endpoint), so the same policy leaves the long replay unchanged. [Tiny group evidence](../04d-gpu/tiny3-dynamic-group8/summary.json), [retained failing long replay](../04d-gpu/long-dynamic-group8-replay/summary.json).

A group-by-group long diagnostic compares every output token against the native 1026-token reference. Groups 0–7 match all outputs exactly. Group8 first fails on the final two-token chunk: 543 values differ, maximum 32. Exact inputs and all eight state buffers before the failed call are saved, allowing one-call replays without rebuilding the prefix. [Locator results](long-group-locator/results.json), [reference](long-group-locator/reference.npz), [inputs](long-group-locator/first-failure-inputs.npz), [state](long-group-locator/first-failure-state-before.npz). Independent audit found no reference or tap error in this locator.

Ordinary group8 replay reproduces the failure. Torch using the actual saved Core ML state matches the native target exactly, excluding earlier state divergence on this fixture. Instrumentation locates the first differing BF16 site at layer34 weighted attention: one element changes from 8.8125 to 8.875. [Stage results](long-group8-stages/results.json), [stage arrays](long-group8-stages/stages.npz).

Controlled full-shape isolation proves that the weighted value multiplication alone causes this discrepancy. Scores and source-exp softmax weights are bitwise exact; the Core ML raw product is 8.84375 versus Torch 8.843749046325684, crossing a BF16 midpoint. Using only the two actual query rows with all 2048 keys fixes both raw and rounded products in this capture. This is bounded compiler arithmetic evidence; an ordinary group replay and complete numerical gates remain required before adopting a query-row policy. [Causal isolation](../04d-gpu/long-layer34/summary.json), [product variants](../04d-gpu/long-layer34/products/summary.json).

The ordinary long group8 correction passes exactly when the value multiplication receives only real query rows; score padding retains min(32, absolute endpoint). Tiny3 also remains exact in the compiled group. The integrated converter independently reproduces the long correction. [Composed long gate](../04d-gpu/long-layer34/group8-productrows/summary.json), [tiny gate](../04d-gpu/tiny3-productrows-group8/summary.json), [integrated-converter long replay](integrated-query-shard/long-replay/results.json).

Full candidate4 is building in its own directory with `--queries dynamic-product`; embedding/head are copied from candidate2. Source snapshots and hashes are captured before conversion. Historical default graph/package selection remains unchanged. The tiny-prefix replay adapter is explicitly opt-in in the CLI and all validation harnesses, with nine focused tests passing; numerical qualification must use this selected adapter. [Build](full-candidate-4/build.json), [adapter audit](../04d-audit/REPORT.md).

The integrated converter also passes tiny3 group8 exactly. Its shard0 regression preserves exact outputs/taps for all three canonical prompts, their chunk remainders and the tested decode steps. This binds the bounded operator corrections to the converter used for candidate4. [Integrated tiny replay](integrated-query-shard/tiny-replay/results.json), [integrated shard0 regression](../04d-gpu/long-layer34/integrated-shard0/gate.json). Full-model acceptance remains pending.

Candidate4 converted all 14 packages and passed build/source/copy provenance verification, aggregate identity `9aaea5e518689705ec236710803e53107a224620e8847004be335e47b7969398`. Its short gate is running with explicit tiny-prefix replay. A misleading query-padding docstring was corrected only after sealing: real query rows come first, then zeros; the executable AST is unchanged and the exact build source snapshot is retained. [Seal](full-candidate-4/provenance-verification.json), [documentation-only change record](full-candidate-4/post-seal-documentation-change.json).

## Candidate4 short gate passes

Candidate4 passes **54/54 numerical comparisons and 17/17 lifecycle checks**, with all 39 hidden comparisons exact, all 15 top1 choices matching, and maximum score error 1.9073486328125e-06 against 0.04. It explicitly selects `tiny_prefix_replay`. Its matched tiny/irregular partition and window/capacity gates are now running. No default promotion or numerical-limit change. [Short summary](full-candidate-4/summary.json), [validation](full-candidate-4/validation-CPU_ONLY.json).

## Extended progress and runtime deadlock

Candidate4 passes window endpoints1023/1024/1025/1026, including the previously failing1026 endpoint; score errors are at most1.9073486328125e-06. Capacity and fresh-partition comparisons are still running. The independent partition run passed its first one-token case, then deadlocked during direct tiny3 before returning a numerical comparison.

Two native samples plus the pinned bridge source establish a GIL/resetQueue lock cycle. The old cleanup guard acquires the GIL for background decref, while synchronous prediction holds that GIL and waits for the same reset queue. Only the stalled audit process was terminated, retaining its completed first case and marking the run `runtime_deadlock`, with numerical qualification incomplete. No numerical failure is inferred from the stall. A separate prediction-boundary guard is under controlled test; package arithmetic remains unchanged. [Runtime report](../04e-runtime/REPORT.md), [partial partition run](../04d-audit/candidate4-partitions/results.json).

The repaired runtime completes the formerly deadlocked directtiny3 call, exposing a full-model numerical failure: score max error0.5823030472, hidden relative RMSE0.0183673. Direct1/15/16 pass. The earlier isolated group8 fix therefore does not qualify the complete tiny-input path; a new native-input group locator is checking earlier/later groups. Capacity2048 is being localized independently. [New partition evidence](../04d-audit/candidate4-runtime2-partitions/results.json).

The prediction-guard full short rerun preserves54/54 numerical and17/17 lifecycle PASS, with39 exact hidden comparisons and max score error1.9073486328125e-06. [Runtime2 short](candidate4-runtime2-short/summary.json). The complete runtime2 partition audit finishes without deadlock:56/97 steps pass,41 fail, all13 requests disposed and the partition repeat exact. Failures remain concentrated at prefixes3–14. [Complete partition results](../04d-audit/candidate4-runtime2-partitions/results.json).

The capacity locator first differs in group0 at positions1058:1090 after earlier chunks match; its ordinary replay reproduces the failure and Torch with the actual saved state matches native reference exactly. The tiny3 locator finds groups0–8 exact and first failure in group9 (layers36–39); Torch from the saved native boundary also matches exactly. Both failures therefore have bounded conversion/runtime-stage probes underway. [Capacity capture](../04d-audit/candidate4-capacity-locator/results.json), [tiny capture](../04d-gpu/candidate4-tiny3-firstgroup/results.json).

## Candidate5 preflight: two independently isolated arithmetic corrections

Capacity failure first occurs at layer0 weighted attention in group0, positions1058:1090. Scores and softmax are exact. Splitting the value product into two-query-row products reproduces all40,960 captured FP32 outputs exactly; groups of4/8/16 and higher-precision accumulation do not. The integrated ordinary group0 fixes the saved failure and retains exact outputs/taps across all nine canonical prompt/remainder/decode steps. [Causal probe](capacity-layer0/summary.json), [shape ablations](capacity-layer0/product-shapes/summary.json), [ordinary capacity replay](paired-group0/capacity-replay/results.json), [canonical gate](paired-group0/gate.json).

Tiny3 first differs at layer38 score multiplication. A sequential128-term FP32 dot over the first16 keys fixes the active scores; subsequent causal masking discards padded inactive scores. A dynamic-shape score package passes both tiny3 and normal32 branches, and an ordinary dynamic four-layer group9 now matches the native tiny3 boundary exactly. The all-group Torch preflight adds no mismatch relative to candidate4, although its pre-existing group8 mismatch still requires compiled verification. [Diagnosis and evidence](../04d-gpu/candidate4-tiny3-score-diagnosis.md), [dynamic group9](../04d-gpu/candidate4-tiny3-group9-dynamic-scores/summary.json).

Both corrections are explicit experimental converter options, `--product paired --score-dot tiny-sequential`, with corresponding builder flags and source snapshot coverage. Combined group8 compiled verification and independent integration review precede full candidate5 conversion. Full short, partition, boundary and generation gates remain necessary; defaults and acceptance limits are unchanged.

The combined group8 preflight **fails**, so candidate5 full conversion is held. Candidate4 compiled group8 is exact on native tiny3 input, while both corrections together produce789 differing BF16 outputs (maximum80), matching the pre-existing Torch discrepancy. This demonstrates an interaction that isolated group0/group9 successes did not cover. Separate tiny-score-only and paired-product-only compiled ablations are running. [Combined regression](../04d-gpu/candidate5-tiny3-group8-combined/summary.json).

Paired-product-only Core ML group8 preserves exact native tiny3 outputs. Source inspection plus layer32 capture shows sequential score reduction is not a general replacement: it disagrees with native3×3 BLAS scores and changes one BF16 attention value. The tiny-score policy is therefore held out. **Full candidate5 is building with paired products only**, retaining candidate4 score arithmetic, to test capacity independently. Tiny-prefix fidelity remains unresolved. [Paired-only group8](../04d-gpu/candidate5-tiny3-group8-paired/summary.json), [candidate5 build](full-candidate-5/build.json).

The isolated score-only Core ML group8 ablation reproduces the789-element regression; the paired-only ablation remains exact. Fresh deterministic BF16 dot probes show sequential accumulation matches native at lengths1/2/15, but disagrees at3/7. Mismatches concentrate in trailing query rows (row2 at3; rows4–6 at7), indicating shape-dependent BLAS behavior. Simple lane folds and full FP64 accumulation do not reproduce all source scores. A two-lane reduction is being tested as an explicitly empirical bounded alternative; it is not selected in candidate5. [Score-only failure](../04d-gpu/candidate5-tiny3-group8-scores-only/summary.json), [fresh reduction matrix](../04d-gpu/bf16-dot-reduction-matrix/summary.json), [shape ablation](../04d-gpu/bf16-dot-shape-ablation/summary.json).

Candidate5 converted all14 packages and passed source/copy provenance verification, aggregate`aced450babb69e87bae9f0cca764d7ea16b883b6f001536c3078ca2ae82af7e2`. Its complete short gate now runs with both guards and explicit tiny-prefix replay. The two-lane tiny-score experiment fails broader full-Torch prefixes despite fixing the saved group8/group9 sites, so it is also held out. [Candidate5 seal](full-candidate-5/provenance-verification.json), [two-lane full-prefix preflight](two-lane-prefix-preflight/results.json).

The broader two-lane full-Torch audit completes63 comparisons across three retained fixture families plus a fresh seeded fixture, prefixes1–16 (one family has15 tokens):10 pass,53 fail, maximum score error1.5667943954. This rejects the proposed general policy despite exact local group8/group9 Core ML replays. [Complete summary](two-lane-prefix-preflight/summary.json). A four-lane alternative also fails early in its broader audit, so no further composed conversion is justified from those local successes.

Candidate5 short gate PASS:54/54 numerical,39 hidden exact,15 top1 matches, maximum score error1.9073486328125e-06,17/17 lifecycle. Both extended gates now run under the same package/runtime identities and tiny-prefix adapter. [Short summary](full-candidate-5/summary.json), [partitions](../04d-audit/candidate5-partitions/results.json), [boundaries](full-candidate-5-boundaries/results.json).

## Candidate5 full capacity correction passes

All6 same-runtime boundary comparisons PASS:1023/1024/1025/1026/2048/fresh1026, maximum score error1.9073486328125e-06. All4 overflow/disposal/partition-repeat checks PASS. The package and runtime provenance maps match the completed54/54 short gate. This closes the recorded2048 capacity numerical failure for this candidate and fixture. [Boundary evidence](full-candidate-5-boundaries/results.json), [combined short/boundary summary](full-candidate-5/summary.json).

The first candidate5 partition attempt records22 steps, including the existing directtiny3 failure, then stops with BrokenPipeError. The harness saves each completed metric before printing progress; the interruption is consistent with its output pipe closing, and is not evidence of a new Core ML numerical or cleanup defect. The raw failed attempt is preserved. A new attempt directs stdout/stderr to a regular log file and retains the same validator/runtime/package identities. [Interrupted audit](../04d-audit/candidate5-partitions/results.json), [rerun](../04d-audit/candidate5-partitions-rerun1/results.json).

The remaining numerical target is tiny-prefix attention. New bounded research will verify the native Accelerate GEMM invocation and characterize its summation behavior; cancellation probes may distinguish reduction orders that generic lane guesses did not recover. Experimental local fixes remain unselected until independent operator and full-prefix gates pass.

The [tiny-prefix operator investigation](../04f-tiny/REPORT.md) now reproduces native raw score dots exactly via source-shaped Accelerate calls on61 above-threshold rectangular cases. This confirms the backend/strides before reduction characterization; it is not yet a Core ML correction.

Candidate5's file-directed partition rerun completes all13 cases and97 steps:56 pass,41 fail,all13 requests disposed,and `[33]` versus `[1,32]` hidden/scores repeat exactly. There is no runtime exception in the rerun. The [same-runtime milestone aggregation](candidate5-milestone.json) verifies provenance,short and boundary gates,but remains incomplete because partitions fail and generation is unrun. [Complete rerun](../04d-audit/candidate5-partitions-rerun1/results.json).

Cancellation-based operator probes recover distinct four-lane summation folds for the three-token trailing query row: key columns0/1 combine `(lane0+lane1)+(lane2+lane3)`, while key column2 combines `(lane0+lane2)+(lane1+lane3)`. Each recovered tree passes1000 held-out topology probes, sign/amplitude controls and saved independent dense operands. Shape coverage and composed model validation are still pending; no new tiny-score policy has been integrated. [New investigation](../04f-tiny/REPORT.md).

## Candidate 6 preflight complete; conversion running

The [native tiny-score follow-up](../04f-tiny/REPORT.md) now passes all 63 whole-Torch prefixes with exact hidden states and scores, the bounded Core ML operator shapes, and both integrated group replays. [Candidate 6 conversion](full-candidate-6/build.json) is running with paired value products and native tiny-score arithmetic. Full gates remain pending.

Candidate6 now has all14 packages and a passing [provenance seal](full-candidate-6/provenance-verification.json), aggregate `5a3e4c7a571e4fc3532377d9b9bb0da023f499c10709216fee7e1001bf4ad2c6`. Full short/lifecycle validation is running.

Candidate6 [short gate](full-candidate-6/summary.json) passes54/54 numerical comparisons and17/17 lifecycle checks. All39 hidden comparisons are exact; all15 top1 comparisons match; maximum score difference is1.9073486328125e-6. Full [partition](../04d-audit/candidate6-partitions/results.json) and [boundary](full-candidate-6-boundaries/results.json) gates are running.

Candidate6 [full partition gate](../04d-audit/candidate6-partitions/summary.json) is complete:13cases,97/97 comparisonsPASS,13disposalsPASS, and the33-token partition repeat has exact hidden states and scores. This resolves all41 recorded candidate5 partition failures. Boundaries andgeneration remain pending.
