# Conditional warm-score qualification

The new arithmetic policy is `conditional-warm-score`. It preserves the
qualified single-model state interface, paired value products and native tiny
scores while conditionally evaluating the short/full score branches inside
each group model. It is a new MIL build, not an unchanged
`paired-native-tiny` conversion. The unchanged persistent compiled loader
remains CPU_ONLY. Any separate GPU diagnosis has a separate candidate and
qualification scope.

`scripts/summarize_conditional_milestone.py` is a read-only aggregator. It
must not run inference or change limits, validators, selected evidence or
existing summaries. Its output path must be new. Current numerical evidence
can only pass when both source/compiled identity verification and the new
conditional build provenance pass.

The selected layout is:

- Source packages: `artifacts/05g-integration/full-1/source`.
- Compiled packages: `artifacts/05g-integration/full-1/compiled`.
- Source/compiled manifests and build evidence:
  `evidence/05g-integration/full-1`.

The baseline evidence is the passing `05c-compiled/milestone-rerun1.json`,
not the preserved failed first aggregation. The aggregator pins the SHA-256
of the passing milestone, cohort source verification, short attempt,
qualified generation, frozen calibration and token/partition references.
Their recorded hashes establish the unchanged source identities; equality
only to today's files is insufficient. This binds the runtime, native
guards, loader, validators, sampling code and qualified reference inputs.

The new source provenance must include source snapshots and hashes for the
new builder; baseline package lineage; all 12 successful per-group structural
records; and the exact new source package identity. Each group has four
rank-0 `less(query_rows,16)` predicates/selectors, with ordinary/output
FP32 shapes `[1,10,Q,2048]`. Each short branch retains its original dynamic
shape computation and appends one identity to refine the result to the
ordinary output shape. Four true branch blocks each contain 2,502 operations
(10,004 original operations moved plus four shape identities); five constants shared between short branches
remain in the parent block. Each false branch contains one identity so its
return is produced locally; total identity count increases by eight. Every surviving original operation, state read
and write, value-product matmul, and non-operation model metadata must be
unchanged. Embedding and head packages must be identical to the baseline.
These record checks are in addition to measuring the final package/cache
bytes, not a replacement for that measurement.

Every gate must report the selected candidate root, matching full source
manifest, `compiled_tiny_prefix_replay`, the same current and qualified
runtime/guard hashes, and the selected compiled manifest path and digest.
The compiled cache must bind the new source identity and pass the existing
strict source/cache byte and environment verification.

The numerical and lifecycle requirements remain:

- Short: 54 passing comparisons and all 17 lifecycle checks; completed
  attempt with matching validator/runtime/loader provenance.
- Partitions: exact frozen set of 13 cases and 97 passing prefix checks;
  13 disposed requests; exact repeated-partition hidden/scores and disposal.
- Boundaries: exact six frozen canonical-cycle comparisons and four passing
  lifecycle/capacity checks.
- Generation: all 674 actual token IDs equal the qualified CPU reference,
  seven correctly labeled score audits, identical fixture/sampling/limits,
  correct boxed 391 after the thinking marker, terminal token 151643 under
  the existing two-EOS policy, `stop_token` finish, and explicit disposal.

The existing CPU aggregator checks counts and status, but does not itself
require token equality. The conditional aggregator adds direct token-list
comparison and rejects tampered token or finish-reason evidence; it does not
weaken any existing gate.

Implementation status: the aggregator CLI and structural/gate checks are ready.
All four frozen qualified gate records pass its evidence-schema checks; direct
negative checks reject a changed generated token and changed finish reason.
The complete conditional candidate still needs its build/seal records and new
qualification results. No conditional candidate is claimed qualified yet.

The final assembly schema is `build.json` with `status=completed`, `passed=true`,
`arithmetic_policy=conditional-warm-score`, the new source `identity`, baseline
`source_candidate_identity`, `compiled_identity`, source/cache roots,
`assembler_sha256`, exactly twelve `{group,record,record_sha256}` entries and
`copied` entries for embedding/head. The compatible seal is
`provenance-verification.json`; its before/after hashes, snapshots, copy lineage
and new source identity are independently checked.

Additional helper checks: synthetic in-memory structural records exercise all
12 groups and reject a false state-preservation flag, missing group, or variant
inventory mismatch. These checks validate aggregator behavior only. The final
CLI still remeasures actual source/cache bytes through the unchanged compiled
identity verifier. No native model prediction or full cache hashing ran during
aggregator development.
