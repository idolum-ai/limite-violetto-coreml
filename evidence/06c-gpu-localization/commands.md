# Commands

All run from the project root with stdout/stderr redirected to matching evidence logs.

```sh
.venv/bin/python scripts/localize_gpu_drift.py --attempt prompt0-1 --protocol evidence/06c-gpu-localization/replay-protocol.json
.venv/bin/python scripts/build_gpu_key_taps.py --attempt key-taps-1 --compile
```

Both completed with exit0. Native calls used authorized sandbox escalation and the existing prediction/cleanup guards.

```sh
.venv/bin/python scripts/replay_gpu_key_taps.py --attempt key-taps-replay-1 --protocol evidence/06c-gpu-localization/key-taps-replay-protocol.json
.venv/bin/python scripts/build_gpu_raw_key_taps.py --attempt raw-key-taps-1 --compile
.venv/bin/python scripts/replay_gpu_raw_key_taps.py --attempt raw-key-replay-1 --protocol evidence/06c-gpu-localization/raw-key-replay-protocol.json
.venv/bin/python scripts/verify_conditional_preservation.py --output evidence/06c-gpu-localization/cpu-preservation.json
```

The first tap replay, raw clone build and CPU preservation check completed exit0; the raw replay subsequently completed exit0 with all controls passing. Both xctrace records used the exact template/instrument options in [placement-plan.md](placement-plan.md), `--time-limit 60s`, and the owned worker `scripts/trace_gpu_replay.py --policy CPU|GPU --attempt placement-cpu-1|placement-gpu-1`. Both recordings ended when the launched worker exited normally, before the limit. Recording logs and worker logs are alongside the trace containers. No other process was attached and no unrelated table was exported. Trace timing is not a performance benchmark.

```sh
.venv/bin/python scripts/summarize_gpu_rounding.py --output evidence/06c-gpu-localization/rounding-summary.json
.venv/bin/python scripts/verify_conditional_preservation.py --output evidence/06c-gpu-localization/cpu-preservation-final.json
python3 scripts/checksum_handoff.py --evidence-dir evidence/06c-gpu-localization
```

Array analysis and final CPU preservation completed exit0. Final checksum output is recorded in checksum-handoff.log and checksum-verification.json.
