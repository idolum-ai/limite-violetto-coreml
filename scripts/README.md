# Research scripts

These scripts retain their original names and locations because they reference one another and the recorded evidence paths. Most are experiment entry points, not a stable public API. Many require the full local archive. Inspect arguments and output paths before execution; some early scripts overwrite fixed evidence files.

| Task | Entry points |
|---|---|
| Safe clone-only checks | `verify_repository.py`; `audit_git_index.py` after staging |
| Portable reference | `validate_portable.py`, plus the `portable.generate` module |
| Core ML conversion | `convert_coreml.py`; later candidate builders are named `build_*` |
| Runtime guards | `build_cleanup_guard.py`, then `build_prediction_guard.py` |
| CPU qualification | `validate_coreml.py`, `audit_coreml_partitions.py`, `validate_coreml_boundaries.py`, `audit_coreml_generation.py` |
| Gate aggregation | `summarize_cpu_milestone.py`, `summarize_compiled_milestone.py`, `summarize_conditional_milestone.py` |
| Compiled caches | `build_compiled_cache.py`; use the exact recorded commands in evidence/05c-compiled |
| GPU diagnostics | `probe_*`, `replay_*`, `analyze_*`; start from the phase report and frozen protocol |
| Focused regression checks | `test_*.py`; run the compiled-identity executable separately |

The [evidence index](../evidence/README.md) is the recommended way to locate an experiment's commands, rather than treating every script as an independent supported workflow. Historical validators can exit normally while recording failed numerical gates. Read the result, not just the exit status.
