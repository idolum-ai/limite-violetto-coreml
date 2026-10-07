# Reproduction commands

Run from `<PROJECT_ROOT>` using the pinned `.venv`. Native compile/load/predict requires normal host access; workers install the pinned process-local guards. These recorded attempt paths already exist: any rerun requires a fresh attempt and a new frozen protocol. Do not overwrite evidence.

```sh
.venv/bin/python scripts/build_normalization_taps.py --attempt taps-1 --compile
.venv/bin/python scripts/probe_normalization_taps.py --attempt replay-1 --build evidence/06e-gpu-normalization/taps-1/results.json
```

Model compilation is a structural/load preparation result. Tap interpretation requires all eight replay calls and all same-policy endpoint/eight-state controls to pass; normal exit alone is insufficient. CPU_AND_GPU is a policy, not measured device placement.

Independent archive and reciprocal analysis (fresh output paths required):

```sh
.venv/bin/python scripts/summarize_rounding_residual.py --replay evidence/06e-gpu-normalization/replay-1/results.json --output evidence/06e-gpu-normalization/residual-summary-1.json
.venv/bin/python scripts/analyze_normalization_drift.py --replay evidence/06e-gpu-normalization/replay-1/results.json --summary evidence/06e-gpu-normalization/residual-summary-1.json --output evidence/06e-gpu-normalization/normalization-analysis.json
.venv/bin/python scripts/verify_conditional_preservation.py --output evidence/06e-gpu-normalization/cpu-preservation.json
```

The offline counterfactual script has a fixed recorded output path and refuses overwrite; a rerun requires a separate snapshot/output selection:

```sh
.venv/bin/python scripts/analyze_normalization_counterfactual.py
```

The correction candidate is **held after host failure**. Its planned native command below was NOT executed; no correction model was built and native prediction count is zero:

```sh
.venv/bin/python scripts/probe_reciprocal_refinement.py --attempt reciprocal-1 --protocol evidence/06e-gpu-normalization/reciprocal-protocol.json
```

Read-only reproduction of the failed host check, without model conversion/loading:

```sh
.venv/bin/python - <<'PY'
import json
from pathlib import Path
from scripts import probe_reciprocal_refinement as probe
protocol = probe.read(Path('evidence/06e-gpu-normalization/reciprocal-protocol.json'))
assert probe.sha(Path(probe.__file__)) == protocol['script_sha256']
assert probe.sha(probe.HELPER) == protocol['helper_sha256']
cases = probe.prepare_cases(probe.read(probe.SOURCE), protocol['source_archives'])
print(json.dumps(probe.host_check(cases), indent=2))
PY
```

After all project/agent writes complete, seal the handoff:

```sh
python3 scripts/checksum_handoff.py --evidence-dir evidence/06e-gpu-normalization > evidence/06e-gpu-normalization/checksum-handoff.log 2>&1
```
