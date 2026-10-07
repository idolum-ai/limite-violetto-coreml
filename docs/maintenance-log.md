# Maintenance decisions

## 2026-10-07 · Initial repository preparation

- Verified the requested remote identity as `idolum-ai/limite-violetto-coreml`. GitHub reported an empty **public** repository; no existing Git refs were found. The requested destination was expected to be private, so no upload or visibility change was performed.
- Prepared the curated checkout separately from the original local archive. Original source/artifact bytes remain local and untouched. No force-push, release, deployment, model conversion, checkpoint download, or inference run was performed.
- Preserved the experiment directory structure and model/runtime code. Replaced the sprawling top-level entry point with concise navigation, a status table, setup instructions, license scope, an archive guide, and a roadmap to the unimplemented scaled reciprocal experiment.
- Exported selected evidence with private filesystem paths redacted and both original/export hashes recorded. Excluded host-bound compiled identities instead of rewriting them into misleading runtime manifests. Historical checksums retain their original meaning.
- Kept licensed pinned source families and existing notices. Excluded ancillary copies with incomplete retained provenance/license material. Introduced no blanket relicensing of original research code or third-party assets.
- Adjusted the CPU milestone unit test to use synthetic runtime identity hashes instead of requiring an excluded native guard binary. Production/runtime/graph code and frozen numerical criteria remain unchanged.
- Added read-only repository and exact-index checks. The initial sandbox blocked `.git` creation in the workspace; an approved filesystem escalation initialized Git and fetched the empty remote without uploading project data.

## 2026-10-07 · Approved public preparation

- The follow-up rubric identified host-local battery IDs and private trace-session paths that the original common-secret/path checks did not cover. It retained research progress, failed experiments, useful provenance and attribution as publication material.
- The owner approved the reviewed narrow cleanup and public publication, including keeping his name. No new license for original contributions was selected; the no-new-license statement in `LICENSE.md` remains unchanged.
- Replaced 58 battery identifier values in 29 JSON exports while preserving power conditions, timing, thermal and result data. Removed 132 private trace entries from each of five checksum lists (660 total entries), with the remaining original hashes unchanged. These lists are now explicitly partial public exports.
- Preserved the original evidence and first local candidate privately; prepared public history from a sanitized root with no unredacted parent. Added private Git bundles to ignore/audit exclusions. Only the approved public branch is eligible for push.
- Extended the scanner for host-local battery identifiers, trace-session references and private trace checksum entries. Four focused integration tests exercise actual Git index contents, including rejection when the working tree has already been cleaned but the staged blob has not.
- Public preparation performs lightweight metadata/integrity and scanner checks. The earlier numerical-library test results remain separately recorded; no model conversion, inference or qualification was rerun.

Validation results are in [validation.md](validation.md). Publication uses a normal push after a fresh remote fetch, preserves concurrent work, and requires verification of the remote commit and any triggered CI. No visibility change, release or deployment is part of this work.
