# Step 3 commands

Working directory: `<PROJECT_ROOT>`. Existing hash-pinned `.venv`; four Torch CPU threads. No packages or services were changed. Metal execution needs host access because the sandbox hides the GPU.

```sh
python3 scripts/download_checkpoint.py > evidence/03-portable/checkpoint-download.log 2>&1
.venv/bin/python scripts/probe_full_model.py mps > evidence/03-portable/calibration-mps.log 2>&1
.venv/bin/python scripts/validate_portable.py --device mps --attempt run1 > evidence/03-portable/run1.log 2>&1
.venv/bin/python scripts/validate_portable.py --device mps --attempt run2 > evidence/03-portable/run2.log 2>&1
.venv/bin/python scripts/probe_full_model.py cpu > evidence/03-portable/calibration-cpu.log 2>&1
.venv/bin/python scripts/diagnose_mps_cache.py > evidence/03-portable/mps-first-layer-diagnosis.log 2>&1
.venv/bin/python scripts/validate_portable.py --device cpu --attempt run3 --max-new-tokens 512 > evidence/03-portable/run3.log 2>&1
.venv/bin/python scripts/check_head_and_stops.py > evidence/03-portable/head-and-stops.log 2>&1
.venv/bin/python -m portable.generate 'Compute 17 * 23.' --device cpu --max-new-tokens 1024 > evidence/03-portable/cli-math.json 2> evidence/03-portable/cli-math.stderr.log
.venv/bin/python -m portable.generate 'Compute 17 * 23.' --device cpu --max-new-tokens 1024 --sample --seed 0 > evidence/03-portable/cli-math-sampled.json 2> evidence/03-portable/cli-math-sampled.stderr.log
.venv/bin/python -m portable.generate 'Compute 17 * 23.' --device cpu --max-new-tokens 2004 --sample --seed 0 > evidence/03-portable/cli-math-full-budget.json 2> evidence/03-portable/cli-math-full-budget.stderr.log
.venv/bin/python -m portable.generate 'Compute 17 * 23.' --device cpu --max-new-tokens 1024 --sample --seed 0 --stop-at-tokenizer-eos > evidence/03-portable/cli-math-explicit-stop.json 2> evidence/03-portable/cli-math-explicit-stop.stderr.log
.venv/bin/python scripts/inspect_generation_stops.py > evidence/03-portable/eos-observation.log
python3 scripts/finalize_step3.py
```

Run1 stopped before cached execution due to the diagnostic FP64/MPS issue. Run2 used the earlier validation script's 192-token generation cap; its code snapshot is at `artifacts/03-portable/run2/code/`. Run3 records the hashes of its scripts in `results.json`. The current runner defaults to a 512-token cap; use explicit arguments for future comparisons.

The ad-hoc CPU cache probe loaded the model on CPU, computed the first 44-token prompt through `reference_hidden`, then cached its first 43 tokens and evaluated the last token. `scripts.probe_full_model.metrics` compared the final hidden state and all head scores. Its JSON/log are retained as `cpu-cache-probe.*`; it was followed by the larger run3 suite, not used as a final acceptance result.

The suite samples are greedy; the separate CLI files record both greedy and seeded sampling. Sampling uses the published 0.6 temperature and 0.95 top-p. The explicit-stop command adds tokenizer EOS151643 while retaining model EOS151645; the default was not changed. The controlled head/stop test was rerun after adding that option; its earlier results are preserved as `before-extra-stop-head-and-stops.*`. Neither CPU nor Metal cached generation has passed the numerical gate. Do not treat plausible text, an EOS, or a passing behavioral test as proof of original-model fidelity.
