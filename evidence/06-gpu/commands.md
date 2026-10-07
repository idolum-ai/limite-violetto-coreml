# Reproduce the GPU-policy diagnostic

Run from the project root using the pinned virtual environment and normal host access. Use a new attempt name; preserve all previous evidence. Keep native experiments sequential.

```sh
.venv/bin/python scripts/probe_gpu_group0.py --attempt gpu-group0-1 > evidence/06-gpu/gpu-group0-1.log 2>&1
```

The script is pinned to the original sealed source and compiled candidate. It snapshots its source and protocol, binds the warm fixtures, saves each output/state array, and verifies identities before and after. A completed inexact comparison is a valid diagnostic outcome with exit0, while a failed control/provenance/guard check invalidates the attempt. Read the exactness and diagnostic fields separately from process status.

Independent saved-array audit (no native inference):

```sh
.venv/bin/python scripts/summarize_gpu_group0.py --results evidence/06-gpu/gpu-group0-1/results.json --output evidence/06-gpu/summary.json
```

Use a new output path when repeating; the script refuses to overwrite an existing summary. The initial audit is preserved as `summary-initial.json`; the final summary adds active-cache-slot metrics without rerunning predictions.
