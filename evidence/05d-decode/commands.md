# Reproduce the profiling milestone

Run from the project root in the pinned virtual environment. Core ML load/predict requires host access. Use new attempt/output names; preserve existing evidence. Keep inference runs sequential.

```sh
.venv/bin/python scripts/profile_compiled_decode.py --attempt profile-1 > evidence/05d-decode/profile-1.log 2>&1
.venv/bin/python scripts/summarize_decode_profile.py --input evidence/05d-decode/profile-1/results.json --output evidence/05d-decode/summary.json > evidence/05d-decode/summary.log 2>&1
.venv/bin/python scripts/capture_decode_group.py --attempt group0-warm-capture --group 0 > evidence/05d-decode/group0-warm-capture.log 2>&1
.venv/bin/python -m unittest scripts.test_decode_profile_summary
```

The static inventory script and invocation are documented in [its report](static-inventory/REPORT.md). It performs package verification and spec reading only; complete that work before a timing run. The next optimization is planned separately in [next-experiment.json](next-experiment.json).
