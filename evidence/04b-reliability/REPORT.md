# Cleanup crash diagnosis — confirmed and mitigated

Authorized by Daniel: diagnose the native crash, then continue; keep the report current and estimate remaining work at multiple finish lines.

## Crash evidence

The original macOS crash report identifies SIGSEGV/invalid address 0x10 on thread 7. The chain is Core ML delayed input-port cleanup → Objective-C destruction → libcoremlpython offset 0x2da44 → CPython `_PyObject_Free`. This locates the crash in Python-owned input cleanup. It is not an out-of-memory termination report.

The installed Core ML Tools 9.0 binary SHA256 is `915f478a3f2a392077012f8dacd9ae433579cbb299019ef2df23217ce536ee67`. Its generated destructor at 0x2da14 decrements a Python reference and calls `_Py_Dealloc` without acquiring the GIL. The exact version's `PybindCompatibleArray` contains a `py::array m_array` member and has no explicit synchronized destructor. The inspected current upstream Array.mm is identical. Source and binary evidence support an unsynchronized Python reference release on a Core ML worker. A deterministic actual-class reproducer is being tested.

A project-local guard wraps only that class's destructor in Python GIL acquisition/release. It checks Python version, coremltools version and the bridge binary hash before installing, and does not modify `.venv` or any Core ML model package. It is a compatibility patch requiring requalification with other versions, not a general Core ML fix. During Python finalization it avoids calling back into the allocator; any remaining reference is left for process teardown.

## Investment estimates at the start of this follow-up

These are judgment-based estimates of effort completed, not measured pass rates or promises:

| Finish line | Estimated done | Remaining |
|---|---:|---|
| Working demonstration with documented limitations | 90–95% | Crash handling and basic benchmark |
| Dependable local runner passing existing numerical limits | 60–70% | Arithmetic correction, reliability regressions, broader tasks |
| Proud-to-release, independently validated and convenient | 35–50% | Independent original-runtime captures, performance, packaging and broader validation |

Numerical fidelity is the largest uncertain cost. No numerical threshold changes are authorized or planned.

## Confirmed reproducer and guard results

A deterministic test creates the actual upstream `PybindCompatibleArray` and releases it on a native dispatch worker. Without the guard, the debug allocator aborts with `_PyMem_DebugFree: Python memory allocator called without holding the GIL` (exit 134). With the guard, all 1,000 scheduled background releases finish successfully, counted explicitly. This reproduces the failure mechanism found in the original macOS stack. The first short wait only observed 551 completed calls and was retained; the accepted test waits for all 1,000.

Full-model guarded CPU validation under the debug allocator passes all 17 lifecycle checks. All 54 numerical comparison records have unchanged metric values relative to the previous unguarded run. The compatibility guard changes cleanup synchronization, not model numerics. Three repetitions of the original long-context call pattern are running separately.

Three complete repetitions of the original 1024-token prefill, two window-crossing comparisons, and fill to 2048 all complete under the debug allocator. All three reject overflow and dispose state correctly; boundary metric values are unchanged. Total wrapper cleanup count after this run is 14,415; these particular calls occurred with the GIL already held. The separate accepted worker-release test proves the missing-GIL path is guarded.

The historical crash is now diagnosed and its specific mechanism is mitigated. This is bounded regression evidence, not a guarantee against every future native/runtime failure. Numerical fidelity remains a separate failing gate.


## Source and binary provenance

- Original crash: [macOS report](original-crash.ips), [parsed stack](crash-stack.json).
- Upstream class: [CoreMLPythonArray.mm at 9.0](https://github.com/apple/coremltools/blob/9.0/coremlpython/CoreMLPythonArray.mm), with the `py::array` member in [its header](https://github.com/apple/coremltools/blob/9.0/coremlpython/CoreMLPythonArray.h). Local copies and hashes are retained.
- [Build record](guard-build.json), [guard source](../../coreml_port/native/cleanup_guard.c), [version/hash checks](../../coreml_port/cleanup_guard.py).
- [Original deterministic failure](destructor-original.log), [all 1000 guarded releases](destructor-guarded.log), [unchanged numerics on both settings](numerical-invariance.json), [three full-context repetitions](long-context-stress.json).
- [Commands](commands.md), [summary](summary.json).

Both complete-model compute settings now pass their 17 lifecycle checks with all 54 numerical comparison records unchanged per setting. G4.3 passes for this recorded guarded scope. G4.2 numerical fidelity is unchanged and remains failing in the deployed experimental packages.

The original crash image UUID matches the installed bridge binary UUID exactly. [Binary identity](binary-identity.json), [destructor disassembly](destructor-disassembly.txt).
