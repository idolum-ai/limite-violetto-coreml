# Reproduce the diagnosed cleanup issue and the mitigation

Run from the project root with the pinned `.venv`. Native code requires the existing Xcode command-line tools. The installed bridge hash is checked at runtime; do not bypass that check after changing Python/coremltools.

```sh
.venv/bin/python scripts/build_cleanup_guard.py
.venv/bin/python scripts/build_cleanup_reproducer.py
# Intentional child-process abort, demonstrating the original upstream defect:
PYTHONMALLOC=debug .venv/bin/python scripts/reproduce_cleanup_destructor.py
# All 1000 background releases must complete:
PYTHONMALLOC=debug .venv/bin/python scripts/reproduce_cleanup_destructor.py --guard --count 1000
# Save new evidence directories on subsequent runs to retain these records:
PYTHONMALLOC=debug .venv/bin/python scripts/validate_coreml.py --evidence-dir evidence/04b-reliability
PYTHONMALLOC=debug .venv/bin/python scripts/stress_guarded_coreml.py
```

`Pipeline` installs the compatibility guard before creating any MLModel objects. It acquires the Python GIL only around the installed bridge class's destructor. It does not patch the operating system, installed wheel, weights, graph operations or scores. The test-only reproducer is never loaded by `Pipeline`.

This patch is local and specific to CPython 3.12 and the recorded Core ML Tools 9.0 bridge binary. The robust upstream solution would implement synchronized ownership cleanup in that bridge class; no upstream message or issue has been sent. A future native Swift runtime could avoid the Python bridge entirely, but is outside this fix.

A 50-call ordinary prediction loop did not reproduce background cleanup. That negative result was retained. The accepted deterministic reproducer invokes the actual upstream wrapper's cleanup on a native worker, matching the captured crash stack; it is not a generic artificial Python crash.
