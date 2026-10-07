# Reproduction commands

Run from the project root using the pinned virtual environment. Native Core ML compilation and prediction require host access. Use unused attempt names; scripts preserve prior results and refuse to replace attempt directories. Keep native experiments sequential.

```sh
.venv/bin/python scripts/capture_product_fixture.py --attempt capture-p128 --prefix 128 > evidence/05e-product/capture-p128.log 2>&1
.venv/bin/python scripts/build_decode_product.py --attempt repeat-first-2 --compile > evidence/05e-product/repeat-first-2.log 2>&1
.venv/bin/python scripts/capture_product_fixture.py --attempt capture-p512 --prefix 512 > evidence/05e-product/capture-p512.log 2>&1
.venv/bin/python scripts/benchmark_decode_product.py --attempt benchmark-1 --protocol evidence/05e-product/protocol.json > evidence/05e-product/benchmark-1.log 2>&1
```

The benchmark protocol pins its script, build result, and three fixture result records; those records bind all NPZ arrays and artifact files. For a new build or capture, create a new protocol file with measured hashes and the same predeclared gates before running. Preserve the old protocol and results.

Build attempt `repeat-first-1` failed in the structural checker before saving or compiling. The historical source snapshot and log are retained. The corrected checker handles zero-output state-write operations without discarding them.

The package itself retains its generic interface to isolate the graph edit. Only the benchmark harness is a supported entry point: it rejects anything outside one query at absolute position 32–2047 before state creation or prediction. There is no full-runtime integration.

Independent aggregation and rejection tests:

```sh
.venv/bin/python scripts/summarize_decode_product.py --results evidence/05e-product/benchmark-1/results.json --protocol evidence/05e-product/protocol.json --output evidence/05e-product/summary.json > evidence/05e-product/summary.log 2>&1
.venv/bin/python -m unittest scripts.test_decode_product_summary > evidence/05e-product/summary-tests.log 2>&1
```

The aggregator refuses to overwrite its output. A valid completed experiment that misses the performance gate exits successfully and records `performance_gate_passed=false`.
