# Completed own-process placement capture

Actual CPU/GPU execution placement remains **unverified**. Both completed traces exported the owned worker's Core ML activity table successfully, but both tables contain their schema and **zero event rows**. This supplies no Prediction coverage and no CPU/GPU dispatch evidence. Zero rows must not be interpreted as zero device use.

| Policy | Owned PID | Worker predictions | Trace Prediction intervals | CPU dispatch intervals | GPU Request intervals |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU_ONLY | 96044 | 2 | 0 | 0 | 0 |
| CPU_AND_GPU | 12316 | 2 | 0 | 0 | 0 |

Each trace TOC identifies exactly one launched `scripts/trace_gpu_replay.py` process, the expected PID, normal exit status 0, and the `coreml-os-signpost` table with `target-pid="SINGLE"`. Only that table was exported. No unrelated event tables were read, no attachment or new recording was performed by the summarizer, and no timing/performance conclusion is drawn.

The worker evidence independently passes: chunks 0/1 each execute one fresh restored-state prediction; outputs and post-state match the corresponding saved policy controls exactly; state references are released; prediction guards count 0 plain calls, 2 state calls, and 2 GIL releases. Fixture, original source/cache cohort, reviewed worker snapshot, and normal process exits are verified. These controls establish that the intended workload ran, but compute policy and numerical replay agreement do not establish actual device placement.

Both `xctrace export` operations report exit 0. Their stderr includes a kernel-extension-info MAC policy error. The evidence does not establish whether that diagnostic caused missing Core ML events; no causal explanation is assumed.

## Reproducible evidence

- [CPU parsed result](placement-cpu-summary-1/results.json), [owned table](placement-cpu-summary-1/coreml-own-pid.xml), [TOC](placement-cpu-summary-1/toc.xml), [worker](placement-cpu-1/results.json).
- [GPU parsed result](placement-gpu-summary-1/results.json), [owned table](placement-gpu-summary-1/coreml-own-pid.xml), [TOC](placement-gpu-summary-1/toc.xml), [worker](placement-gpu-1/results.json).
- [Parser](../../scripts/summarize_gpu_placement.py), snapshot copied into both summary directories; SHA-256 `6e0a06e8df20e75988e5467073bdccc2e9fd0f6663a9231fea170075c188ca85`.
- [Installed-instrument interpretation and limits](placement-plan.md).

The parser reports `status="completed"` for successful scoped export/interpretation and separately `actual_placement_verified=false`. This is an explicit evidence limit, not a placement qualification pass. Further profiling is outside this bounded attempt.
