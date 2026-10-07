# Step 4 commands and interpretation

Run from `<PROJECT_ROOT>`, using the existing locked `.venv` on this Mac. Core ML host execution requires access outside the execution sandbox. These scripts write results to `evidence/04-coreml`; preserve the existing attempt before rerunning. Exported packages are under `artifacts/04-coreml/model`.

```sh
.venv/bin/python experiments/probe_bf16_rounding.py
.venv/bin/python scripts/convert_coreml.py 0
.venv/bin/python scripts/convert_coreml_all.py
.venv/bin/python scripts/probe_coreml_shard.py
.venv/bin/python scripts/probe_coreml_graph.py
.venv/bin/python scripts/probe_coreml_precision.py
.venv/bin/python scripts/validate_coreml.py
.venv/bin/python scripts/validate_coreml.py --compute CPU_AND_GPU
.venv/bin/python scripts/probe_coreml_boundary.py
.venv/bin/python scripts/probe_coreml_longprefill.py
.venv/bin/python -m coreml_port.generate 'What is 17 * 23?' --max-new-tokens 1024 --sample --seed 0 --stop-at-tokenizer-eos --stream
```

The rounding probe has retained first and second attempts; its existing directory deliberately prevents overwriting. The conversion batch covers embedding, groups 1–11, and head; group 0 is converted first for the shard experiment. Keep the hash-pinned input checkpoint, tokenizer and configuration unchanged.

The default complete-model runtime uses `CPU_ONLY`. `--compute CPU_AND_GPU` permits GPU scheduling; it does not establish actual placement. Streaming text goes to stderr, the final JSON record to stdout. Ctrl-C releases request state and exits 130. Each generation request owns twelve state objects; no request cache is shared. Context is bounded to 2048, prefill chunks to 32, and padded vocabulary rows are rejected/excluded.

The default stop token remains published actor EOS151645. `--stop-at-tokenizer-eos` explicitly adds EOS151643; the result records the policy. Do not equate a process exit code of zero with numerical qualification: inspect `numerical_pass`, `lifecycle_pass`, exact metrics, and the final report. All neural computation is inside the Core ML packages; host operations perform orchestration, tokenization, sampling and text decoding. This is an unquantized weight export but uses explicit BF16 rounding emulation and FP16 state, and is not an exact native-BF16 implementation.

The same-input generation comparison uses `.venv/bin/python scripts/compare_coreml_generation.py` after the recorded Core ML generation JSON exists. It saves the complete qualified CPU reference result, exact input/seed comparison, and first token divergence.

The recorded runs overlap local work. Timings are diagnostic, not a controlled benchmark or throughput claim.
