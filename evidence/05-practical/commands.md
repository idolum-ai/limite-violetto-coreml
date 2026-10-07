# Practical pilot reproduction

Run from `<PROJECT_ROOT>` using the existing pinned `.venv`. The candidate and runtime must still match the passing candidate6 milestone. Core ML needs host framework access and installs both pinned process-local guards. The host-observation commands may also require access outside the filesystem sandbox.

Run the backends **sequentially**, each in a fresh process. Do not run the task suite or other agent inference concurrently. Every output directory below is already occupied by this recorded run; choose new names for reproduction. Keep the same benchmark script for both backends.

```sh
.venv/bin/python scripts/benchmark_practical_cpu.py --backend coreml --evidence-dir evidence/05-practical/coreml-pilot-1 > evidence/05-practical/coreml-pilot-1.log 2>&1
.venv/bin/python scripts/benchmark_practical_cpu.py --backend torch --evidence-dir evidence/05-practical/torch-pilot-1 > evidence/05-practical/torch-pilot-1.log 2>&1
.venv/bin/python scripts/summarize_practical_cpu.py --coreml evidence/05-practical/coreml-pilot-1/results.json --torch evidence/05-practical/torch-pilot-1/results.json --output evidence/05-practical/comparison.json
.venv/bin/python scripts/evaluate_practical_tasks.py --self-check
.venv/bin/python scripts/evaluate_practical_tasks.py --attempt cpu-reference-1 > evidence/05-practical/cpu-reference-1.log 2>&1
```

The benchmark retains one first-use request and three trials per shape. The comparison treats repeat0 as first-shape use and only repeats1–2 as warm reuse. It checks all27 saved score vectors against the same frozen numerical limits outside the timed spans. A performance ratio is descriptive; no universal usefulness threshold or cold-cache startup claim is made.

The task suite uses the portable CPU reference, not Core ML. Task response timing includes progress persistence and is not the benchmark. Its parser requires completed responses with an unambiguous numeric final boxed answer after the reasoning terminator. The original Core ML 674-token answer remains separate qualified evidence.
