"""Isolated whole-pipeline ABBA comparison after conditional correctness passes.

One fresh worker owns one CPU_ONLY pipeline at a time. Source/cache verification,
loading, state creation/disposal, saving and equality checks are outside timers.
No runtime, guard, arithmetic, sampling, or cache policy is modified.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
FIXTURE = "evidence/04-coreml/generation-comparison.json"
FIXTURE_SHA256 = "1ee6553885123a57b8fb40b89c292eea201e148e6f3bd4d9b4cba1c057f9f3eb"
BASELINE_SOURCE = "5a3e4c7a571e4fc3532377d9b9bb0da023f499c10709216fee7e1001bf4ad2c6"
BASELINE_COMPILED = "a838dd36e8f87c6b9fc0dd7865d0912651c6ab3399fa299d10ff12d788bcb031"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def resolve(value):
    path=Path(value).expanduser()
    return (path if path.is_absolute() else ROOT/path).resolve()


def read(path):
    value=json.loads(Path(path).read_text())
    if not isinstance(value,dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def save(path,record):
    temporary=path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record,indent=2,sort_keys=True)+"\n")
    temporary.replace(path)


def host():
    return {"utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
            "platform":platform.platform(),"load_average":list(os.getloadavg()),
            "power":subprocess.run(["pmset","-g","batt"],capture_output=True,text=True).stdout,
            "thermal":subprocess.run(["pmset","-g","therm"],capture_output=True,text=True).stdout}


def validate_protocol(protocol):
    expected={"schema_version":1,"fixture":FIXTURE,"fixture_sha256":FIXTURE_SHA256,
              "arm_order":"ABBA","round_prefix_orders":[[32,128,512],[512,128,32]],
              "decode_steps":3,"warmup_prefix":32,"warmup_decode_steps":3,
              "minimum_median_decode_reduction":0.10,"maximum_median_prefill_regression":0.10,
              "compute":"CPU_ONLY","runtime_adapter":"compiled_tiny_prefix_replay"}
    for key,value in expected.items():
        if protocol.get(key)!=value:
            raise ValueError(f"Predeclared protocol differs: {key}")
    if protocol.get("script_sha256")!=sha(__file__):
        raise ValueError("Benchmark source differs from sealed protocol")
    helper=ROOT/"scripts/summarize_conditional_milestone.py"
    if protocol.get("qualification_helper_sha256")!=sha(helper):
        raise ValueError("Qualification helper differs from sealed protocol")
    if sha(ROOT/FIXTURE)!=FIXTURE_SHA256:
        raise ValueError("Teacher-forced fixture differs")
    if set(protocol.get("arms",{}))!={"A","B"}:
        raise ValueError("Expected explicit baseline A and conditional B")
    for arm,selection in protocol["arms"].items():
        if set(selection)!={"model_dir","identity_manifest","compiled_dir","compiled_manifest"}:
            raise ValueError(f"Incomplete explicit cohort selection: {arm}")
    return protocol


def bind_cohort(protocol,candidate_milestone):
    from scripts.summarize_conditional_milestone import qualified_context,POLICY
    context=qualified_context()
    if context["errors"]:
        raise ValueError(context["errors"])
    milestone=read(candidate_milestone)
    if milestone.get("milestone_pass") is not True or milestone.get("status")!="completed" or milestone.get("selected_arithmetic_policy")!=POLICY:
        raise ValueError("Conditional candidate must first pass its complete correctness milestone")
    if milestone.get("thresholds_changed") is not False or milestone.get("compute")!="CPU_ONLY":
        raise ValueError("Candidate qualification scope differs")
    if milestone.get("aggregator_sha256")!=protocol["qualification_helper_sha256"]:
        raise ValueError("Candidate milestone used a different qualification aggregator")
    expected_sources=context["expected_sources"]
    if milestone.get("qualified_source_sha256")!=expected_sources:
        raise ValueError("Candidate milestone source cohort differs")
    identities={}
    for arm,selection in protocol["arms"].items():
        manifest_path=resolve(selection["identity_manifest"])
        compiled_path=resolve(selection["compiled_manifest"])
        source,compiled=read(manifest_path),read(compiled_path)
        identities[arm]={"source":source["aggregate_sha256"],"compiled":compiled["aggregate_sha256"],
                         "source_manifest_sha256":sha(manifest_path),"compiled_manifest_sha256":sha(compiled_path),
                         "selection":{k:str(resolve(v)) for k,v in selection.items()}}
        if compiled.get("source_candidate_identity")!=source["aggregate_sha256"]:
            raise ValueError(f"Source/cache manifest binding mismatch: {arm}")
    if identities["A"]["source"]!=BASELINE_SOURCE or identities["A"]["compiled"]!=BASELINE_COMPILED:
        raise ValueError("Baseline is not the qualified source/cache cohort")
    source=milestone["source_provenance"];compiled=milestone["compiled_identity"]
    if source.get("passed") is not True or compiled.get("passed") is not True:
        raise ValueError("Candidate identities did not pass qualification")
    b=identities["B"]
    if source.get("source_package_aggregate_sha256")!=b["source"] or source.get("manifest_sha256")!=b["source_manifest_sha256"]:
        raise ValueError("Candidate source manifest differs from passing milestone")
    loader=compiled["loader_provenance"]
    if loader.get("compiled_aggregate_sha256")!=b["compiled"] or loader.get("compiled_manifest_sha256")!=b["compiled_manifest_sha256"]:
        raise ValueError("Candidate compiled manifest differs from passing milestone")
    if source.get("candidate_root")!=b["selection"]["model_dir"] or loader.get("compiled_dir")!=b["selection"]["compiled_dir"] or loader.get("compiled_manifest")!=b["selection"]["compiled_manifest"]:
        raise ValueError("Candidate paths differ from passing milestone")
    # Gates are not rerun here; immutable evidence file hashes must still agree.
    for gate in milestone["gates"].values():
        if gate.get("passed") is not True or sha(resolve(gate["path"]))!=gate["result_sha256"]:
            raise ValueError("Qualified candidate gate evidence changed")
        if "attempt_path" in gate and sha(resolve(gate["attempt_path"]))!=gate["attempt_sha256"]:
            raise ValueError("Qualified candidate attempt evidence changed")
    return {"identities":identities,"qualified_sources":expected_sources,
            "candidate_milestone":str(candidate_milestone),"candidate_milestone_sha256":sha(candidate_milestone),
            "runtime_provenance":context["milestone"]["current_runtime_provenance"]}


def summarize(rows,protocol):
    cells=[]
    for prefix in (32,128,512):
        cell={"prefix":prefix,"sample_counts":{}}
        for phase in ("prefill","decode"):
            values={arm:[r["seconds"] for r in rows if r["prefix"]==prefix and r["phase"]==phase and r["arm"]==arm] for arm in "AB"}
            expected=4 if phase=="prefill" else 12
            if any(len(v)!=expected or any(not isinstance(x,(int,float)) or not 0<x<float("inf") for x in v) for v in values.values()):
                raise ValueError(f"Incomplete or invalid timed samples: {prefix}/{phase}")
            med={arm:statistics.median(v) for arm,v in values.items()}
            cell[phase]={"baseline_median_seconds":med["A"],"conditional_median_seconds":med["B"],
                         "conditional_over_baseline":med["B"]/med["A"],"median_reduction":1-med["B"]/med["A"]}
            cell["sample_counts"][phase]={arm:len(v) for arm,v in values.items()}
        cell["decode_go"]=cell["decode"]["conditional_over_baseline"]<=1-protocol["minimum_median_decode_reduction"]
        cell["prefill_go"]=cell["prefill"]["conditional_over_baseline"]<=1+protocol["maximum_median_prefill_regression"]
        cells.append(cell)
    med={arm:statistics.median(r["seconds"] for r in rows if r["arm"]==arm and r["phase"]=="decode") for arm in "AB"}
    pooled=1-med["B"]/med["A"]
    rounds=[]
    for rnd in range(2):
        for prefix in (32,128,512):
            medians={arm:statistics.median(r["seconds"] for r in rows if r["round"]==rnd and r["prefix"]==prefix and r["phase"]=="decode" and r["arm"]==arm) for arm in "AB"}
            rounds.append({"round":rnd,"prefix":prefix,"median_reduction":1-medians["B"]/medians["A"]})
    unstable=any(abs(rounds[p]["median_reduction"]-rounds[p+3]["median_reduction"])>0.10 for p in range(3))
    return {"prefixes":cells,"pooled_median_decode_reduction":pooled,"round_prefix_decode_reductions":rounds,
            "round_reduction_spread_over_10_percentage_points":unstable,
            "performance_gate_passed":med["B"]/med["A"]<=1-protocol["minimum_median_decode_reduction"] and all(c["decode_go"] and c["prefill_go"] for c in cells),
            "interpretation":"Observed local timings; only two ABBA rounds, uncontrolled OS cache and host scheduling. Round spread and host telemetry must accompany the result."}


def worker(args,protocol):
    import numpy as np
    import torch
    torch.set_num_threads(4)
    from coreml_port.loader_selection import create_pipeline,loader_provenance
    from coreml_port.prediction_guard import stats
    from coreml_port.cleanup_guard import stats as cleanup_stats
    from scripts.runtime_provenance import runtime_provenance
    out=resolve(args.worker_output);out.mkdir(parents=True,exist_ok=False)
    record={"status":"starting","arm":args.arm,"round":args.round,"slot":args.slot,
            "script_sha256":sha(__file__),"protocol_sha256":sha(resolve(args.protocol)),"timings":[],"requests":[]}
    save(out/"results.json",record)
    arrays={};model=None
    try:
        cohort=bind_cohort(protocol,resolve(args.candidate_milestone));record["cohort"]=cohort
        selection=protocol["arms"][args.arm]
        loader_args=argparse.Namespace(**{k:str(resolve(v)) for k,v in selection.items()},compute="CPU_ONLY",tiny_prefix_replay=True)
        record["loader_provenance"]=loader_provenance(loader_args);record["runtime_provenance"]=runtime_provenance()
        started=time.perf_counter();model=create_pipeline(loader_args);record["initialization_seconds"]=time.perf_counter()-started
        record["identity_verification_seconds"]=model.identity_verification_seconds
        record["component_load_seconds"]=model.load_seconds;record["source_spec_seconds"]=model.source_spec_seconds
        identity=cohort["identities"][args.arm]
        if model.package_identity["aggregate_sha256"]!=identity["source"] or model.compiled_identity["aggregate_sha256"]!=identity["compiled"]:
            raise ValueError("Loaded pipeline identities differ")
        ref=read(ROOT/FIXTURE)["reference"];stream=ref["prompt_ids"]+ref["generated_ids"]
        record["forced_stream_sha256"]=hashlib.sha256(json.dumps(stream,separators=(",",":")).encode()).hexdigest()
        def request(prefix,warmup=False):
            start=time.perf_counter();req=model.request();created=time.perf_counter()-start
            info={"prefix":prefix,"warmup":warmup,"creation_seconds":created}
            try:
                for step in range(4):
                    ids=stream[:prefix] if step==0 else [stream[prefix+step-1]]
                    started=time.perf_counter_ns();scores=model.forward(ids,req);seconds=(time.perf_counter_ns()-started)/1e9
                    if not warmup:
                        record["timings"].append({"prefix":prefix,"step":step,"phase":"prefill" if step==0 else "decode",
                                                  "seconds":seconds,"arm":args.arm,"round":args.round,"slot":args.slot})
                        arrays[f"p{prefix}_s{step}"]=np.array(scores,copy=True)
                if req.length!=prefix+3:
                    raise ValueError("Teacher-forced request length differs")
            finally:
                started=time.perf_counter();req.close();info["disposal_seconds"]=time.perf_counter()-started
                info["disposed"]=req.closed and not req.states
                record["requests"].append(info)
                if not info["disposed"]:
                    raise ValueError("Request was not disposed")
        request(32,warmup=True)
        record["host_before_timing"]=host()
        for prefix in protocol["round_prefix_orders"][args.round]:
            request(prefix);save(out/"results.json",record)
        record["host_after_timing"]=host()
        record["prediction_guard"]=stats();record["cleanup_guard"]=cleanup_stats()
        if record["prediction_guard"]!={"installed":True,"plain_calls":50,"state_calls":408,"gil_releases":458}:
            raise ValueError("Unexpected guarded prediction counts")
        np.savez(out/"scores.npz",**arrays);record["scores_sha256"]=sha(out/"scores.npz")
        if runtime_provenance()!=record["runtime_provenance"] or bind_cohort(protocol,resolve(args.candidate_milestone))!=cohort:
            raise ValueError("Cohort/runtime evidence changed during worker")
        record.update(status="completed",passed=True)
    except BaseException as exc:
        record.update(status="failed",passed=False,error={"type":type(exc).__name__,"message":str(exc)})
        raise
    finally:
        del model;gc.collect();save(out/"results.json",record)
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol",required=True);p.add_argument("--candidate-milestone",required=True)
    p.add_argument("--attempt");p.add_argument("--confirm-isolated",action="store_true",help="Assert no other model, GPU, or benchmark workload is running.")
    p.add_argument("--worker-output",help=argparse.SUPPRESS);p.add_argument("--arm",choices=list("AB"),help=argparse.SUPPRESS)
    p.add_argument("--round",type=int,choices=[0,1],help=argparse.SUPPRESS);p.add_argument("--slot",type=int,choices=range(4),help=argparse.SUPPRESS)
    args=p.parse_args();protocol=validate_protocol(read(resolve(args.protocol)))
    if args.worker_output:
        return worker(args,protocol)
    if not args.confirm_isolated:
        p.error("Run only after other native workloads finish; pass --confirm-isolated.")
    if not args.attempt or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in args.attempt):
        p.error("A fresh lowercase path-safe --attempt is required")
    cohort=bind_cohort(protocol,resolve(args.candidate_milestone))
    out=ROOT/"evidence/05g-integration"/args.attempt;out.mkdir(parents=True,exist_ok=False)
    (out/"source-snapshot.py").write_bytes(Path(__file__).read_bytes())
    (out/"protocol.json").write_bytes(resolve(args.protocol).read_bytes())
    record={"status":"starting","script_sha256":sha(__file__),"protocol_sha256":sha(resolve(args.protocol)),
            "protocol":protocol,"cohort":cohort,"workers":[],"timings":[],"score_comparisons":[],
            "isolation":"Operator asserted no concurrent model/GPU/benchmark workload; workers run sequentially with one full pipeline resident.",
            "timing_boundary":"Synchronous full Pipeline.forward only. Prefill includes existing 32-token chunking; load/verification, request creation/disposal, scores saving/equality and host telemetry excluded.",
            "host_before":host()}
    save(out/"results.json",record)
    try:
        import numpy as np
        expected=None
        for rnd in range(2):
            for slot,arm in enumerate("ABBA"):
                worker_out=out/f"round{rnd}-slot{slot}-{arm}"
                command=[sys.executable,str(Path(__file__).resolve()),"--protocol",str(resolve(args.protocol)),
                         "--candidate-milestone",str(resolve(args.candidate_milestone)),"--worker-output",str(worker_out),
                         "--arm",arm,"--round",str(rnd),"--slot",str(slot)]
                log=out/f"round{rnd}-slot{slot}-{arm}.log"
                print(f"starting round {rnd} slot {slot} arm {arm}",flush=True)
                with log.open("wb") as output:
                    process=subprocess.run(command,stdout=output,stderr=subprocess.STDOUT)
                if process.returncode:
                    raise RuntimeError(f"Worker {rnd}/{slot}/{arm} exited {process.returncode}; see {log}")
                result=read(worker_out/"results.json")
                if result.get("passed") is not True or result.get("cohort")!=cohort:
                    raise ValueError("Worker failed or changed cohort")
                score_file=worker_out/"scores.npz"
                if sha(score_file)!=result["scores_sha256"]:
                    raise ValueError("Worker score archive identity differs")
                with np.load(score_file) as data:
                    scores={key:data[key].copy() for key in data.files}
                keys={f"p{prefix}_s{step}" for prefix in (32,128,512) for step in range(4)}
                if set(scores)!=keys:
                    raise ValueError("Worker score site set differs")
                if expected is None:
                    expected=scores
                exact={key:bool(np.array_equal(scores[key],expected[key])) for key in sorted(keys)}
                record["score_comparisons"].append({"round":rnd,"slot":slot,"arm":arm,"exact":exact})
                if not all(exact.values()):
                    raise ValueError("Bitwise score parity failed; no performance decision")
                record["workers"].append({"round":rnd,"slot":slot,"arm":arm,"result":str(worker_out/"results.json"),
                                          "result_sha256":sha(worker_out/"results.json"),"scores_sha256":result["scores_sha256"],"exit_code":process.returncode})
                record["timings"].extend(result["timings"]);save(out/"results.json",record)
        record["performance"]=summarize(record["timings"],protocol)
        record["host_after"]=host()
        if bind_cohort(protocol,resolve(args.candidate_milestone))!=cohort or sha(__file__)!=record["script_sha256"] or sha(resolve(args.protocol))!=record["protocol_sha256"]:
            raise ValueError("Benchmark inputs changed during run")
        record.update(status="completed",experiment_completed=True,all_scores_exact=True,
                      decision="performance go; review host noise before promotion" if record["performance"]["performance_gate_passed"] else "performance no-go; do not promote")
    except BaseException as exc:
        record.update(status="failed",experiment_completed=False,decision="no promotion",error={"type":type(exc).__name__,"message":str(exc)})
        raise
    finally:
        save(out/"results.json",record)
    print(json.dumps({"status":record["status"],"decision":record["decision"],"performance":record["performance"]},indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
