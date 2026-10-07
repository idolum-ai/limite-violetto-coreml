# Cache follow-up commands

Working directory: `<PROJECT_ROOT>`; existing pinned `.venv`, Torch CPU threads set to four by each runner. CPU tests run in the sandbox; Metal tests needed host access. No dependencies, upstream files, services, or remote resources were changed.

Recorded ablations:

```sh
.venv/bin/python scripts/probe_cache_reductions.py --device cpu --attempt cpu1
.venv/bin/python scripts/probe_cache_reductions.py --device cpu --attempt cpu-window --modes query_pad --long
.venv/bin/python scripts/probe_cache_reductions.py --device mps --attempt mps-window --modes native,query_pad --long
.venv/bin/python scripts/probe_cache_reductions.py --device cpu --attempt cpu-small-rows --modes rows_4,rows_8,rows_16,rows_32
```

Recorded final validation:

```sh
.venv/bin/python scripts/validate_portable.py --device cpu --phase 03b-cache --attempt cpu-sixteen --max-new-tokens 1
.venv/bin/python scripts/stress_cpu_cache.py
.venv/bin/python scripts/check_cache_warmup.py
.venv/bin/python scripts/audit_cpu_generation.py
python3 scripts/finalize_cache_followup.py
```

Logs are the corresponding `.log` files in this folder. The original `cpu-gates` run used eight rows before warmup; `cpu-final` used eight rows plus warmup. `cpu-sixteen` is the accepted final sixteen-row implementation. The live report and operator trail record the implementation transitions and failed extra tests. Code snapshots are in `artifacts/03b-cache/`; final run JSON files include relevant hashes.

`probe_cache_reductions.py` explicitly disables the production minimum-row policy before applying each isolated ablation. The FP64 mode is a changed-precision diagnostic and is not the production implementation.

The two small inline diagnoses are retained as `prefix-shape-probe.*` and `tiny17-diagnosis.*`: with seed 913 and random valid token IDs, the first compares unmodified upstream prefix lengths 1/2/3/4/7/8/9/16/32 to a 127-token full prefix and then tests cached extension; the second compares original prefixes 16 and 17 and tests minimum query rows 8/16/32/full-prefix for the 16→17 cache transition. Their JSON records the measured quantities.

For a repeat, use an unused `--attempt` label. Scripts with fixed output names require archiving or selecting new output paths before rerunning; the generation auditor refuses to overwrite its result. Do not overwrite the historical failures. The finalizer asserts that the final suite, extended stress, lifecycle checks and generation audit all pass before updating the manifest.

Run the user-facing CPU example without touching evidence files:

```sh
.venv/bin/python -m portable.generate 'Compute 17 * 23.' --device cpu --max-new-tokens 1024 --sample --seed 0 --stop-at-tokenizer-eos
```

This explicitly adds EOS151643 to the published EOS151645 stop set. The unmodified actor config remains pinned; the related value-model documentation and local emissions supporting the extra stop are cited in the report.
