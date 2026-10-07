# Reproduction commands

Run from the project root in the pinned environment. Fresh attempt and output paths are required for reruns. Native compile/load/predict requires normal host Core ML access; guard installation precedes those operations.

```sh
.venv/bin/python scripts/verify_conditional_preservation.py --output evidence/06g-gpu-residual/cpu-preservation-before.json
.venv/bin/python scripts/build_residual_norm_taps.py --preflight
.venv/bin/python scripts/build_residual_norm_taps.py --attempt residual-taps-1 --compile
.venv/bin/python scripts/probe_residual_norm_taps.py --build evidence/06g-gpu-residual/residual-taps-1/results.json --attempt replay-1 --protocol evidence/06g-gpu-residual/replay-1-protocol.json
.venv/bin/python scripts/analyze_residual_norm.py
```

The replay protocol binds source files and all input/expected archives before execution. A normal exit alone does not pass numerical controls; inspect the result and independent analysis.

Separate downstream interval follow-up (frozen after first probe's negative finding):

```sh
.venv/bin/python scripts/build_residual_interval_taps.py --preflight
.venv/bin/python scripts/build_residual_interval_taps.py --attempt interval-taps-1 --compile
.venv/bin/python scripts/probe_residual_interval_taps.py --build evidence/06g-gpu-residual/interval-taps-1/results.json --attempt interval-replay-1 --protocol evidence/06g-gpu-residual/interval-replay-1-protocol.json
.venv/bin/python scripts/analyze_residual_interval.py
```

Final layer2 key-path probe (frozen after interval localization):

```sh
.venv/bin/python scripts/build_residual_key_taps.py --preflight
.venv/bin/python scripts/build_residual_key_taps.py --attempt key-taps-1 --compile
.venv/bin/python scripts/probe_residual_key_taps.py --build evidence/06g-gpu-residual/key-taps-1/results.json --attempt key-replay-1 --protocol evidence/06g-gpu-residual/key-replay-1-protocol.json
.venv/bin/python scripts/analyze_residual_key.py
.venv/bin/python scripts/verify_conditional_preservation.py --output evidence/06g-gpu-residual/cpu-preservation.json
```

Controlled saved-tensor substitution and final preservation/seal:

```sh
.venv/bin/python scripts/analyze_key_reciprocal_counterfactual.py
.venv/bin/python scripts/verify_conditional_preservation.py --output evidence/06g-gpu-residual/cpu-preservation-final.json
python3 scripts/checksum_handoff.py --evidence-dir evidence/06g-gpu-residual > evidence/06g-gpu-residual/checksum-handoff.log 2>&1
```

The checksum command is the final project write. Read checksum-verification.json for its authoritative completion result.
