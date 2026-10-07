# Own-process runtime placement capture plan

Installed tooling supports a promising **actual execution-event** path. It is
not yet validated on this model: no trace recording, model loading, prediction,
attachment, or inspection of another process was performed for this plan.

## Verified local capabilities

`/Applications/Xcode.app/Contents/Developer/usr/bin/xctrace` reports version
`16.0 (17F113)`. Installed templates include `Core ML` and `Metal System Trace`;
instruments include `Core ML`, `Metal Application`, `GPU`, `Neural Engine`,
`os_signpost`, and `stdout/stderr`. Local `record` help supports an explicit
`--launch -- command arguments`, `--instrument`, `--time-limit`, run names,
output paths and launched-target stdout redirection. `export` supports a table
of contents and XPath-selected XML tables.

More usefully, Apple's installed Core ML instrument implementation describes
runtime dispatch intervals rather than planned device assignments. The package
root is:

```
/Applications/Xcode.app/Contents/Applications/Instruments.app/Contents/Packages/CoreML.instrdst/Contents
```

The schema `Extensions/com.apple.coreml.dtac/schemas.xml` defines
`coreml-os-signpost` with required `target-pid`. Its exported fields include
`start`, `duration`, `event-label`, `model-name`, `fully-qualified-model`,
`parent-model-name`, `model-level`, `event-type`, and `graph-event-label`.
The installed documentation says the target attribute limits rows to all
processes or one PID. The actual recorder in
`Extensions/com.apple.modeler-coreml-os-signpost.dtac/modeling-rules/rules-0000.clp`
(lines 1405–1409, 1445–1449, 1499–1503) rejects records whose process is not in
`target-pid` before emitting rows.

Within that same modeler directory:

- `rules-0004.clp` creates `event-label = CPU`,
  `event-type = model-dispatch-event` from matched classic CPU or BNNS
  synchronous begin/end tracepoints. It ties them to the Core ML model stream
  and process.
- `rules-0003.clp` creates `event-label = GPU Request`,
  `event-type = model-dispatch-event`. The start comes from the first matching
  `IOGPU_KernelSegmentBegin`; the end comes from completion of the last command
  buffer for the model workload. It carries model, parent-model and process
  attribution. This is stronger execution evidence than an async submission
  or a compute-plan preference.
- The same GPU modeler separately emits `Prepare GPU Request` as
  `model-data-event`. **Do not count this as GPU execution.**
- `rules-0000.clp` emits `Prediction` for model prediction intervals. Compile,
  load, input copies and similar events are not device-dispatch evidence.

The built-in Core ML template bundles additional instruments, including GPU,
Metal Application, a CPU sampler, Neural Engine and thermal tracks. Prefer a
minimal template plus only the Core ML instrument for the first capture:

```
/Applications/Xcode.app/Contents/Applications/Instruments.app/Contents/Packages/Base.instrdst/Contents/Templates/Blank.tracetemplate
```

Its keyed-archive string inventory contains no instrument IDs. This avoids
intentionally requesting unrelated system/device tracks. The combination of
that template and `--instrument 'Core ML'` is supported by local CLI options
but has **not yet been recording-tested**.

## Proposed bounded experiment

Use an explicitly launched new worker owned by this experiment, with one
verified original group-0 compiled model resident. Do not attach to an existing
Python/application process, use `--all-processes`, or inspect other processes.
A minimal launcher worker still needs to be implemented/coordinated with the
replay owner; the command below is a template, not an existing runnable worker.

Run two separate launches sequentially: `CPU_ONLY` control, then
`CPU_AND_GPU`. Each worker should:

1. Record its PID, script/protocol hashes, source/compiled identities, fixture
   and input/state hashes, compute-policy argument and guard installation.
2. Load only the original group-0 model before its two bounded calls. Restore
   the exact captured input/state pair independently for each call: the cold
   32-token chunk and warm 12-token chunk from the frozen failed 44-token prompt.
   Use the same saved inputs/states across device-policy launches. Do not add
   instrumented intermediate outputs in this first placement capture.
3. Record which of the two calls begins/ends, monotonic/wall timestamps,
   output/state hashes, normal completion and disposal. Preserve the numeric
   replay comparison as a separate result. No sampling, full generation,
   model-dependent tokens or unrelated work belongs in the trace worker.
4. Exit normally after exactly two stateful predictions. Keep initialization
   and prediction events distinguishable. The worker must have its own bounded
   lifecycle: the CLI recording limit by itself is not documented to kill a
   stuck launched process.

Suggested initial cap: 60 seconds per trace, covering one-group loading and
two predictions. If initialization consumes the cap, classify the trace as
incomplete; do not infer missing execution devices. Root can choose a larger
explicit bounded cap before a new attempt if measurements justify it.

A third control is unnecessary unless the first actual trace demonstrates a
missing field or unsupported event path. If native profiling permission or
recording fails, preserve the error and mark placement unverified rather than
changing host profiling/security settings automatically.

## Launch and export command templates

Do not execute until the dedicated worker and its exact arguments are frozen.
The strings `OWNED_WORKER.py` and `OWNED_EXPLICIT_ARGUMENTS` below are placeholders.
Use absolute paths for the interpreter, worker and selected artifacts.

```sh
/Applications/Xcode.app/Contents/Developer/usr/bin/xctrace record \
  --template '/Applications/Xcode.app/Contents/Applications/Instruments.app/Contents/Packages/Base.instrdst/Contents/Templates/Blank.tracetemplate' \
  --instrument 'Core ML' \
  --run-name 'violetto-group0-CPU_AND_GPU-two-calls' \
  --time-limit 60s \
  --output <PROJECT_ROOT>/evidence/06c-gpu-localization/placement-gpu-1.trace \
  --target-stdout <PROJECT_ROOT>/evidence/06c-gpu-localization/placement-gpu-1-worker.log \
  --launch -- <PROJECT_ROOT>/.venv/bin/python \
  <PROJECT_ROOT>/scripts/OWNED_WORKER.py OWNED_EXPLICIT_ARGUMENTS

/Applications/Xcode.app/Contents/Developer/usr/bin/xctrace export \
  --input <PROJECT_ROOT>/evidence/06c-gpu-localization/placement-gpu-1.trace \
  --toc \
  --output <PROJECT_ROOT>/evidence/06c-gpu-localization/placement-gpu-1-toc.xml
```

Read only the run/target metadata and table descriptions needed to verify the
new worker PID, then select the **own-PID** `coreml-os-signpost` table. Do not
assume an attribute's exported spelling/value before inspecting the TOC. The
installed help's XPath form is:

```sh
/Applications/Xcode.app/Contents/Developer/usr/bin/xctrace export \
  --input <PROJECT_ROOT>/evidence/06c-gpu-localization/placement-gpu-1.trace \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="coreml-os-signpost"]' \
  --output <PROJECT_ROOT>/evidence/06c-gpu-localization/placement-gpu-1-coreml.xml
```

Use that expression only if the TOC shows exactly one such table and its target
scope is the launched PID. Otherwise add the exact target-attribute predicate
from the TOC. Reject an ALL/unattributed table; do not broaden the export to
other processes. A `SINGLE` scope needs the run's explicit target PID to match
the worker's own PID. Repeat with distinct CPU control paths and run name.

## Parse and report actual observations

The concrete XML row representation must be inspected after the first trace;
no trace exists yet. Parse schema columns by mnemonic rather than relying on a
fixed column position. Resolve XML `id`/`ref` cells and retain both formatted and
raw values. Preserve raw XML and its digest with the tool/modeler identities.

Require the two expected completed `Prediction` intervals for the selected
model and own target. Determine any top/sub-model hierarchy from the exported
labels; do not assume two top-level rows before inspecting the event inventory.
Associate model dispatch intervals with those calls by the single-model
hierarchy and ordered non-overlapping prediction windows. If association is
ambiguous, report model/run-level placement only.

Count positive, complete `model-dispatch-event` rows labeled `CPU` and
`GPU Request`; retain their start, duration, model and parent-model. Other
labels must first be interpreted against the installed modeler. Report:

- Actual CPU dispatch observed: yes/no/insufficient trace.
- Actual GPU request execution observed: yes/no/insufficient trace.
- For each of the two calls: CPU/GPU interval counts and matched model labels.
- Any incomplete/missing prediction coverage, ambiguous target or hierarchy,
  and all unrecognized dispatch labels.

Positive own-target `GPU Request` intervals constitute evidence of GPU
execution for that captured model workload. They do not establish that every
operation ran on GPU. CPU events establish CPU work in the same limited sense.
Absence is not proof of absence without coverage; missing event families,
modeler incompatibility, truncation or lost events can hide work. The installed
schema has no public operation-ID column, so operation-by-operation placement
is **not established** by this plan.

Do not sum overlapping/nested CPU and GPU durations into utilization fractions.
The GPU interval spans a workload from first kernel segment to final command
completion and may include gaps; it is not pure hardware busy time. Async
submission, `Prepare GPU Request`, model compilation, estimated device cost,
`MLComputePlan` preferences and `CPU_AND_GPU` policy are not substitutes for
execution events. Instrumented timings are not benchmark timings.

The Core ML modeler consumes selected kernel-event classes and signposts, then
filters exported model rows by target PID. Installed help/source verifies that
output scope; it does not prove that the raw trace container contains zero
incidental OS/kernel metadata about other activity. Do not inspect or export
unrelated raw tables. If the requirement is literally zero unrelated raw
collection, that stricter guarantee remains unverified with this tooling.

## Installed-source identities

These small installed files were read and hashed; no model artifact was changed.
Paths below are relative to the CoreML package root unless stated otherwise.

| Source | SHA-256 |
| --- | --- |
| `Extensions/com.apple.coreml.dtac/schemas.xml` | `6300f595959d4715fc3099ee764c715e371b2ec19e21dffb0e75c55845e10059` |
| `Extensions/com.apple.modeler-coreml-os-signpost.dtac/modeling-rules/rules-0000.clp` | `7aab35f7f1aeb669c115dec1dc5d1ebe68fbd06fc0de784211feff28753745b4` |
| `Extensions/com.apple.modeler-coreml-os-signpost.dtac/modeling-rules/rules-0003.clp` | `3935270e8cf5b0e62603f639690c53b150b8c225c15c96b47943c5b6394a8f48` |
| `Extensions/com.apple.modeler-coreml-os-signpost.dtac/modeling-rules/rules-0004.clp` | `3ae67dc1f951b30cd3c097b41c044a8192dac79a2effee6c3c05fd54e56f404b` |
| Sibling `Base.instrdst/Contents/Templates/Blank.tracetemplate` | `18bf55346d5f0154fa9bb0c5640151e28dc430ede074d1a2e924259218756a6b` |

Status: installed capability and runtime-event semantics verified locally;
actual capture/export on the frozen model and actual observed placement remain
unverified pending the explicitly scoped recording experiment.
