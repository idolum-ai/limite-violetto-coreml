# Reproduction commands

Run from the project root in the pinned virtual environment, with host Core ML access. Each output name must be new; completed attempts are immutable. Commands below record the completed cohort; the report links each passing gate and the preserved reporting failure. Keep startup probes sequential and free from concurrent inference. Qualification timings are not benchmarks.

```sh
.venv/bin/python scripts/build_compiled_cache.py --attempt full-cache-1 --source-root artifacts/04d-cpu/full-candidate-6 --source-identity evidence/04d-cpu/full-candidate-6/package-identity.json > evidence/05c-compiled/full-cache-1.log 2>&1
.venv/bin/python scripts/probe_compiled_pipeline.py --model-dir artifacts/04d-cpu/full-candidate-6 --identity-manifest evidence/04d-cpu/full-candidate-6/package-identity.json --compiled-dir artifacts/05c-compiled/full-cache-1 --compiled-manifest evidence/05c-compiled/full-cache-1/compiled-identity.json --evidence-dir evidence/05c-compiled/compiled-first > evidence/05c-compiled/compiled-first.log 2>&1
.venv/bin/python scripts/probe_compiled_pipeline.py --model-dir artifacts/04d-cpu/full-candidate-6 --identity-manifest evidence/04d-cpu/full-candidate-6/package-identity.json --evidence-dir evidence/05c-compiled/package-control > evidence/05c-compiled/package-control.log 2>&1
.venv/bin/python scripts/probe_compiled_pipeline.py --model-dir artifacts/04d-cpu/full-candidate-6 --identity-manifest evidence/04d-cpu/full-candidate-6/package-identity.json --compiled-dir artifacts/05c-compiled/full-cache-1 --compiled-manifest evidence/05c-compiled/full-cache-1/compiled-identity.json --evidence-dir evidence/05c-compiled/compiled-reopen1 > evidence/05c-compiled/compiled-reopen1.log 2>&1
.venv/bin/python scripts/probe_compiled_pipeline.py --model-dir artifacts/04d-cpu/full-candidate-6 --identity-manifest evidence/04d-cpu/full-candidate-6/package-identity.json --compiled-dir artifacts/05c-compiled/full-cache-1 --compiled-manifest evidence/05c-compiled/full-cache-1/compiled-identity.json --evidence-dir evidence/05c-compiled/compiled-reopen2 > evidence/05c-compiled/compiled-reopen2.log 2>&1
.venv/bin/python scripts/validate_coreml.py --model-dir artifacts/04d-cpu/full-candidate-6 --identity-manifest evidence/04d-cpu/full-candidate-6/package-identity.json --compiled-dir artifacts/05c-compiled/full-cache-1 --compiled-manifest evidence/05c-compiled/full-cache-1/compiled-identity.json --tiny-prefix-replay --evidence-dir evidence/05c-compiled/short > evidence/05c-compiled/short.log 2>&1
.venv/bin/python scripts/audit_coreml_partitions.py --model-dir artifacts/04d-cpu/full-candidate-6 --identity-manifest evidence/04d-cpu/full-candidate-6/package-identity.json --compiled-dir artifacts/05c-compiled/full-cache-1 --compiled-manifest evidence/05c-compiled/full-cache-1/compiled-identity.json --tiny-prefix-replay --evidence-dir evidence/04d-audit/compiled1-partitions > evidence/04d-audit/compiled1-partitions.log 2>&1
.venv/bin/python scripts/validate_coreml_boundaries.py --model-dir artifacts/04d-cpu/full-candidate-6 --identity-manifest evidence/04d-cpu/full-candidate-6/package-identity.json --compiled-dir artifacts/05c-compiled/full-cache-1 --compiled-manifest evidence/05c-compiled/full-cache-1/compiled-identity.json --tiny-prefix-replay --evidence-dir evidence/05c-compiled/boundaries > evidence/05c-compiled/boundaries.log 2>&1
.venv/bin/python scripts/audit_coreml_generation.py --model-dir artifacts/04d-cpu/full-candidate-6 --identity-manifest evidence/04d-cpu/full-candidate-6/package-identity.json --compiled-dir artifacts/05c-compiled/full-cache-1 --compiled-manifest evidence/05c-compiled/full-cache-1/compiled-identity.json --tiny-prefix-replay --evidence-dir evidence/04d-audit/compiled1-generation > evidence/04d-audit/compiled1-generation.log 2>&1
```

Additional completed startup aggregation and CLI smoke (two-token budget intentionally produces an incomplete response):

```sh
.venv/bin/python scripts/summarize_compiled_startup.py > evidence/05c-compiled/startup-summary.log 2>&1
.venv/bin/python -m coreml_port.compiled_generate 'What is 17 * 23?' --model-dir artifacts/04d-cpu/full-candidate-6 --identity-manifest evidence/04d-cpu/full-candidate-6/package-identity.json --compiled-dir artifacts/05c-compiled/full-cache-1 --compiled-manifest evidence/05c-compiled/full-cache-1/compiled-identity.json --max-new-tokens 2 --sample --seed 0 --stop-at-tokenizer-eos --stream > evidence/05c-compiled/cli-smoke.json 2> evidence/05c-compiled/cli-smoke.log
```

Final aggregation (run only after all four gates finish; refuses to overwrite its output):

```sh
.venv/bin/python scripts/summarize_compiled_milestone.py --candidate-root artifacts/04d-cpu/full-candidate-6 --candidate-evidence-dir evidence/04d-cpu/full-candidate-6 --compiled-dir artifacts/05c-compiled/full-cache-1 --compiled-manifest evidence/05c-compiled/full-cache-1/compiled-identity.json --short-result evidence/05c-compiled/short/validation-CPU_ONLY.json --short-attempt evidence/05c-compiled/short/validation-CPU_ONLY-attempt.json --partitions-result evidence/04d-audit/compiled1-partitions/results.json --boundaries-result evidence/05c-compiled/boundaries/results.json --generation-result evidence/04d-audit/compiled1-generation/results.json --arithmetic-policy paired-native-tiny --output evidence/05c-compiled/milestone.json > evidence/05c-compiled/milestone.log 2>&1
```

The first aggregate above failed on a reporting argument error (source-code hashes instead of the package manifest). Its JSON/log/source are preserved. After correcting only that argument, the completed rerun passed:

```sh
.venv/bin/python scripts/summarize_compiled_milestone.py --candidate-root artifacts/04d-cpu/full-candidate-6 --candidate-evidence-dir evidence/04d-cpu/full-candidate-6 --compiled-dir artifacts/05c-compiled/full-cache-1 --compiled-manifest evidence/05c-compiled/full-cache-1/compiled-identity.json --short-result evidence/05c-compiled/short/validation-CPU_ONLY.json --short-attempt evidence/05c-compiled/short/validation-CPU_ONLY-attempt.json --partitions-result evidence/04d-audit/compiled1-partitions/results.json --boundaries-result evidence/05c-compiled/boundaries/results.json --generation-result evidence/04d-audit/compiled1-generation/results.json --arithmetic-policy paired-native-tiny --output evidence/05c-compiled/milestone-rerun1.json > evidence/05c-compiled/milestone-rerun1.log 2>&1
```

The runtime, validators, model artifacts and recorded inference results were unchanged between aggregate attempts.
