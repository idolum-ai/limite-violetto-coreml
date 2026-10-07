# Reproduction commands

Run from `<PROJECT_ROOT>` in the pinned environment. Attempt paths are immutable; reruns require a fresh path and frozen matching protocol. Native compile/load/predict commands need normal host access. Existing guards install before native work.

```sh
.venv/bin/python scripts/preflight_boundary_reciprocal.py --attempt host-1
.venv/bin/python scripts/probe_boundary_reciprocal.py --attempt native-1 --protocol evidence/06f-gpu-reciprocal/native-1-protocol.json
```

Native-1 completes36 calls. C3b supplied-seed arithmetic and CPU preservation PASS; C3 saved baseline reproduction FAIL. Its top-level `passed=false` correctly retains that distinction; do not claim a model repair from the isolated experiment.

Single-site group build and controlled replay:

```sh
.venv/bin/python scripts/build_boundary_reciprocal_group.py --attempt group0-1 --compile
.venv/bin/python scripts/probe_boundary_reciprocal_group.py --attempt group-replay-1 --build evidence/06f-gpu-reciprocal/group0-1/results.json --protocol evidence/06f-gpu-reciprocal/group-replay-1-protocol.json
.venv/bin/python scripts/verify_conditional_preservation.py --output evidence/06f-gpu-reciprocal/cpu-preservation.json
```

The group replay completes 12 stateful calls with all local controls and correction gates passing. The standalone C3 failure remains retained. Independent summary and final handoff:

```sh
.venv/bin/python scripts/summarize_boundary_reciprocal_group.py --result evidence/06f-gpu-reciprocal/group-replay-1/results.json --output evidence/06f-gpu-reciprocal/group-summary.json
python3 scripts/checksum_handoff.py --evidence-dir evidence/06f-gpu-reciprocal > evidence/06f-gpu-reciprocal/checksum-handoff.log 2>&1
```

The checksum command is the final project write; its JSON result records completion. No whole-model GPU qualification or runtime promotion was performed.
