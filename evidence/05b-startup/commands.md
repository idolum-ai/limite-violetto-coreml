# Reproduce the isolated startup experiment

Run from `<PROJECT_ROOT>` in the pinned `.venv`, with host Core ML access. The probe installs both qualified process-local guards, verifies the original group package/runtime, and creates new side-by-side artifacts. No pipeline files are changed.

```sh
.venv/bin/python -m py_compile scripts/probe_persistent_compiled.py
.venv/bin/python scripts/probe_persistent_compiled.py --attempt group0-1 > evidence/05b-startup/group0-1.log 2>&1
```

`group0-1` is occupied by the completed record. Use a new unique attempt name to rerun, retaining the same protocol. A new compiled path will have its own first opening; do not treat its results as cold-machine or cache-evicted measurements. The parent launches compile, six interleaved fresh-process load/state workers and two fresh-process prediction workers. It checks native output equality, repeated independent restored states, all cross-loader output/cache arrays and immutable source/compiled artifacts.

The parent JSON retains every command, child exit status and process wall duration. Child JSON separates constructor, state creation, setup and captured prediction times. Large stdout/stderr streams are redirected to regular per-child log files. A passing exit requires every child and cross-loader/identity check to pass.

The package source is `artifacts/04d-cpu/full-candidate-6/0.mlpackage`, bound to the sealed candidate aggregate. The replay inputs/states are retained under `evidence/04d-audit/candidate4-capacity-locator/`; their hashes are recorded by each worker. The retained `expected` tensor is native reference output, not the old failed candidate's `actual` tensor.
