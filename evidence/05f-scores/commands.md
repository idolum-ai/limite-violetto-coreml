# Reproduce the isolated CPU experiment

Use the pinned environment, host access for native execution, unused attempt paths, and sequential native runs.

```sh
.venv/bin/python scripts/build_warm_scores.py --attempt warm-only-1 --compile > evidence/05f-scores/warm-only-1.log 2>&1
.venv/bin/python scripts/benchmark_warm_scores.py --attempt benchmark-1 --protocol evidence/05f-scores/protocol.json > evidence/05f-scores/benchmark-1.log 2>&1
.venv/bin/python scripts/summarize_warm_scores.py --results evidence/05f-scores/benchmark-1/results.json --protocol evidence/05f-scores/protocol.json --output evidence/05f-scores/summary.json > evidence/05f-scores/summary.log 2>&1
```

For new attempts, freeze a new protocol that pins the new build result and benchmark source while retaining the same gates. The benchmark and summarizer are copies of the 05e tools with only their evidence path, benchmark identity path and descriptive docstrings changed. Existing eight rejection tests were also run against the 05f summary function; [test log](summary-tests.log).
