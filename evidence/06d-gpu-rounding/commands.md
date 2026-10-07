# 06d reproduction commands

Run from `<PROJECT_ROOT>` with the pinned `.venv`. Native compile/load/predict commands require normal host access. Each worker installs the pinned native guards. Recorded attempt names below already exist and are immutable: a rerun needs fresh names and, where required, a separately frozen protocol with matching attempt and source hashes. Logs sit beside each attempt directory.

```sh
.venv/bin/python scripts/probe_signed_even_round.py --attempt operator-1 --protocol evidence/06d-gpu-rounding/operator-protocol.json
.venv/bin/python scripts/probe_signed_even_round_v2.py --attempt operator-2 --protocol evidence/06d-gpu-rounding/operator-2-protocol.json
.venv/bin/python scripts/probe_signed_zero_path.py --attempt zero-path-1 --protocol evidence/06d-gpu-rounding/zero-path-protocol.json
.venv/bin/python scripts/probe_signed_zero_path_v2.py --attempt zero-path-2 --protocol evidence/06d-gpu-rounding/zero-path-2-protocol.json
.venv/bin/python scripts/probe_native_original_quantizer.py --attempt operator-baseline-1 --protocol evidence/06d-gpu-rounding/operator-baseline-protocol.json
.venv/bin/python scripts/probe_round_replacement.py --attempt round-replacement-1 --protocol evidence/06d-gpu-rounding/round-replacement-protocol.json
.venv/bin/python scripts/build_signed_even_group_v3.py --attempt group0-1 --compile
.venv/bin/python scripts/probe_signed_even_group.py --attempt group0-replay-1 --build evidence/06d-gpu-rounding/group0-1/results.json
.venv/bin/python scripts/summarize_rounding_group.py
.venv/bin/python scripts/build_rounding_residual_taps.py --attempt residual-taps-1 --compile
.venv/bin/python scripts/probe_rounding_residual_taps.py --attempt residual-replay-1 --build evidence/06d-gpu-rounding/residual-taps-1/results.json
.venv/bin/python scripts/verify_conditional_preservation.py --output evidence/06d-gpu-rounding/cpu-preservation.json
```

A normal exit is not a numerical PASS. The first two candidates and native baseline fail the strict full-quantizer Torch-byte gate identically at signed zero. The v3 operation gate and native CPU preservation pass separately; the uninstrumented GPU group exactness gate still fails. The first zero-path probe fails conversion, and the seven-tap residual probe fails GPU-policy model loading; retain both failures.

The reduced four-tap attempt completes all eight native controls:

```sh
.venv/bin/python scripts/build_rounding_residual_taps_v2.py --attempt residual-taps-2 --compile
.venv/bin/python scripts/probe_rounding_residual_taps.py --attempt residual-replay-2 --build evidence/06d-gpu-rounding/residual-taps-2/results.json
```

The six-tap follow-up separates the attention residual and MLP:

```sh
.venv/bin/python scripts/build_rounding_residual_taps_v3.py --attempt residual-taps-3 --compile
.venv/bin/python scripts/probe_rounding_residual_taps.py --attempt residual-replay-3 --build evidence/06d-gpu-rounding/residual-taps-3/results.json
```

Independent residual archive audits (fresh output paths required):

```sh
.venv/bin/python scripts/summarize_rounding_residual.py --replay evidence/06d-gpu-rounding/residual-replay-2/results.json --output evidence/06d-gpu-rounding/residual-summary-2.json
.venv/bin/python scripts/summarize_rounding_residual.py --replay evidence/06d-gpu-rounding/residual-replay-3/results.json --output evidence/06d-gpu-rounding/residual-summary-3.json
```

After all project and agent writes finish, seal the handoff:

```sh
python3 scripts/checksum_handoff.py --evidence-dir evidence/06d-gpu-rounding > evidence/06d-gpu-rounding/checksum-handoff.log 2>&1
```
