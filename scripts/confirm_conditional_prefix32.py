"""One bounded, independently frozen P32 ABBA confirmation; no retries.

Preserves the original benchmark and uses its pinned cohort/host helpers.
Every worker warms 32+3, then measures a fresh request's 32+3 only.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
HELPER="scripts/benchmark_conditional_pipeline.py"
HELPER_SHA256="8e1e1288205dfaeef2420dc02841139c09478902700b599608b9fe1d95db0485"


def sha(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def resolve(value):
    p=Path(value).expanduser();return (p if p.is_absolute() else ROOT/p).resolve()


def read(path):
    v=json.loads(Path(path).read_text())
    if not isinstance(v,dict):raise ValueError(f"Expected JSON object: {path}")
    return v


def save(path,value):
    temp=path.with_suffix(".tmp");temp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n");temp.replace(path)


def validate(protocol,milestone_path):
    if sha(ROOT/HELPER)!=HELPER_SHA256 or protocol.get("helper")!=HELPER or protocol.get("helper_sha256")!=HELPER_SHA256:
        raise ValueError("Frozen original benchmark helper changed")
    from scripts import benchmark_conditional_pipeline as helper
    expected={"schema_version":1,"attempt":"prefix32-confirmation-1","prefix":32,"arm_order":"ABBA",
              "warmup_decode_steps":3,"measured_decode_steps":3,"minimum_median_decode_reduction":0.10,
              "maximum_median_prefill_regression":0.10,"retry_policy":"one attempt; a failure or no-go ends confirmation this turn",
              "host_stability_gate":"all worker timing boundaries report one recognized unchanged AC/Battery source and standard negative thermal/performance warning messages"}
    for key,value in expected.items():
        if protocol.get(key)!=value:raise ValueError(f"Frozen confirmation plan changed: {key}")
    if protocol.get("script_sha256")!=sha(__file__):raise ValueError("Confirmation source changed")
    for field in ("original_protocol","original_results","historical_record","historical_scores"):
        if sha(resolve(protocol[field]))!=protocol[field+"_sha256"]:raise ValueError(f"Pinned evidence changed: {field}")
    if sha(milestone_path)!=protocol["candidate_milestone_sha256"]:raise ValueError("Passing candidate milestone changed")
    original=helper.validate_protocol(read(resolve(protocol["original_protocol"])))
    cohort=helper.bind_cohort(original,milestone_path)
    prior=read(resolve(protocol["original_results"]))
    if prior.get("status")!="completed" or prior.get("all_scores_exact") is not True or prior.get("cohort")!=cohort:
        raise ValueError("Original completed exact benchmark cohort differs")
    if prior.get("performance",{}).get("performance_gate_passed") is not True:
        raise ValueError("Confirmation requires the original overall performance go")
    historical=read(resolve(protocol["historical_record"]))
    if historical.get("status")!="completed" or historical.get("candidate_identity")!=helper.BASELINE_SOURCE or historical.get("fixture_sha256")!=helper.FIXTURE_SHA256 or historical.get("scores_sha256")!=protocol["historical_scores_sha256"]:
        raise ValueError("Historical P32 fixture provenance differs")
    return helper,original,cohort


def snapshots(out,protocol_path):
    files={"source-snapshot.py":Path(__file__),"protocol.json":protocol_path,"benchmark-helper.py":ROOT/HELPER}
    for name,path in files.items():(out/name).write_bytes(path.read_bytes())
    return {name:sha(out/name) for name in files}


def bitwise_equal(actual,expected):
    return actual.shape==expected.shape and actual.dtype==expected.dtype and actual.tobytes(order="C")==expected.tobytes(order="C")


def host_stability(samples):
    """Fail closed on observed power changes or recorded/missing thermal status."""
    recognized={"Now drawing from 'AC Power'":"AC", "Now drawing from 'Battery Power'":"Battery"}
    required={"Note: No thermal warning level has been recorded", "Note: No performance warning level has been recorded"}
    allowed=required|{"Note: No CPU power status has been recorded"}
    sources=[];checks=[];errors=[]
    for sample in samples:
        power=sample.get("power","")
        first=power.splitlines()[0].strip() if isinstance(power,str) and power.splitlines() else ""
        source=recognized.get(first)
        thermal=sample.get("thermal","")
        lines={line.strip() for line in thermal.splitlines() if line.strip()} if isinstance(thermal,str) else set()
        clear=required<=lines and lines<=allowed
        checks.append({"power_first_line":first,"recognized_source":source,"thermal_performance_clear":clear,"thermal":thermal})
        if source is None:errors.append("missing or unrecognized power-source first line")
        else:sources.append(source)
        if not clear:errors.append("thermal/performance negative status missing or warning/nonstandard status recorded")
    if not samples:errors.append("host timing samples are missing")
    if len(set(sources))>1:errors.append("observed power-source transition across timing boundaries")
    return {"passed":not errors,"power_source":sources[0] if sources and len(set(sources))==1 else None,"sample_count":len(samples),"checks":checks,"errors":errors}


def summarize(rows,protocol):
    result={}
    for phase in ("prefill","decode"):
        samples={arm:[r["seconds"] for r in rows if r["arm"]==arm and r["phase"]==phase] for arm in "AB"}
        expected=2 if phase=="prefill" else 6
        if any(len(values)!=expected or any(not isinstance(v,(int,float)) or not 0<v<float("inf") for v in values) for values in samples.values()):
            raise ValueError(f"Incomplete or invalid {phase} samples")
        med={arm:statistics.median(v) for arm,v in samples.items()}
        result[phase]={"sample_count_per_arm":expected,"baseline_median_seconds":med["A"],
                       "conditional_median_seconds":med["B"],"conditional_over_baseline":med["B"]/med["A"],
                       "median_reduction":1-med["B"]/med["A"]}
    result["performance_gate_passed"]=(result["decode"]["conditional_over_baseline"]<=1-protocol["minimum_median_decode_reduction"] and result["prefill"]["conditional_over_baseline"]<=1+protocol["maximum_median_prefill_regression"])
    result["interpretation"]="One predeclared ABBA confirmation; two prefills and six decode observations per arm. Host telemetry and uncontrolled OS caches remain relevant. No further retry this turn."
    return result


def worker(args,protocol,helper,original,cohort):
    import numpy as np
    import torch
    torch.set_num_threads(4)
    from coreml_port.loader_selection import create_pipeline,loader_provenance
    from coreml_port.prediction_guard import stats
    from coreml_port.cleanup_guard import stats as cleanup_stats
    from scripts.runtime_provenance import runtime_provenance
    if args.arm!="ABBA"[args.slot]:raise ValueError("Worker arm/slot differs from ABBA")
    out=resolve(args.worker_output);out.mkdir(parents=True,exist_ok=False)
    record={"status":"starting","arm":args.arm,"slot":args.slot,"script_sha256":sha(__file__),
            "protocol_sha256":sha(resolve(args.protocol)),"snapshots":snapshots(out,resolve(args.protocol)),
            "cohort":cohort,"requests":[],"timings":[]}
    save(out/"results.json",record);model=None;arrays={}
    try:
        selection=original["arms"][args.arm]
        loader_args=argparse.Namespace(**{k:str(resolve(v)) for k,v in selection.items()},compute="CPU_ONLY",tiny_prefix_replay=True)
        record["loader_provenance"]=loader_provenance(loader_args);record["runtime_provenance"]=runtime_provenance()
        start=time.perf_counter();model=create_pipeline(loader_args);record["initialization_seconds"]=time.perf_counter()-start
        record["identity_verification_seconds"]=model.identity_verification_seconds
        record["component_load_seconds"]=model.load_seconds;record["source_spec_seconds"]=model.source_spec_seconds
        bound=cohort["identities"][args.arm]
        if model.package_identity["aggregate_sha256"]!=bound["source"] or model.compiled_identity["aggregate_sha256"]!=bound["compiled"]:
            raise ValueError("Loaded source/cache cohort differs")
        ref=read(ROOT/helper.FIXTURE)["reference"];stream=ref["prompt_ids"]+ref["generated_ids"]
        record["forced_stream_sha256"]=hashlib.sha256(json.dumps(stream,separators=(",",":")).encode()).hexdigest()
        def request(warmup):
            start=time.perf_counter();req=model.request();info={"warmup":warmup,"creation_seconds":time.perf_counter()-start}
            try:
                for step in range(4):
                    ids=stream[:32] if step==0 else [stream[32+step-1]]
                    start=time.perf_counter_ns();scores=model.forward(ids,req);elapsed=(time.perf_counter_ns()-start)/1e9
                    if not warmup:
                        record["timings"].append({"arm":args.arm,"slot":args.slot,"prefix":32,"step":step,
                                                  "phase":"prefill" if step==0 else "decode","seconds":elapsed})
                        arrays[f"p32_s{step}"]=np.array(scores,copy=True)
                if req.length!=35:raise ValueError("Teacher-forced request length differs")
            finally:
                start=time.perf_counter();req.close();info["disposal_seconds"]=time.perf_counter()-start
                info["disposed"]=req.closed and not req.states;record["requests"].append(info)
                if not info["disposed"]:raise ValueError("Request disposal failed")
        request(True);record["host_before_timing"]=helper.host()
        record["host_stability"]=host_stability([record["host_before_timing"]])
        if not record["host_stability"]["passed"]:raise ValueError("Host stability precondition failed; no timing or retry")
        request(False);record["host_after_timing"]=helper.host()
        record["host_stability"]=host_stability([record["host_before_timing"],record["host_after_timing"]])
        if not record["host_stability"]["passed"]:raise ValueError("Power/thermal stability failed during worker; no promotion or retry")
        record["prediction_guard"]=stats();record["cleanup_guard"]=cleanup_stats()
        if record["prediction_guard"]!={"installed":True,"plain_calls":16,"state_calls":96,"gil_releases":112}:
            raise ValueError("Prediction guard counts differ from frozen P32 plan")
        with np.load(resolve(protocol["historical_scores"])) as historical:
            exact={key:bitwise_equal(value,historical[key]) for key,value in arrays.items()}
        record["historical_scores_bitwise_exact"]=exact
        if len(exact)!=4 or not all(exact.values()):raise ValueError("Scores differ from pinned historical P32 fixture")
        np.savez(out/"scores.npz",**arrays);record["scores_sha256"]=sha(out/"scores.npz")
        if runtime_provenance()!=record["runtime_provenance"] or validate(protocol,resolve(args.candidate_milestone))[2]!=cohort:
            raise ValueError("Cohort/runtime changed during confirmation worker")
        if sha(resolve(args.protocol))!=record["protocol_sha256"]:raise ValueError("Confirmation protocol changed during worker")
        record.update(status="completed",passed=True)
    except BaseException as exc:
        record.update(status="failed",passed=False,error={"type":type(exc).__name__,"message":str(exc)});raise
    finally:
        del model;gc.collect();save(out/"results.json",record)
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--attempt");p.add_argument("--protocol",required=True);p.add_argument("--candidate-milestone",required=True)
    p.add_argument("--confirm-isolated",action="store_true")
    p.add_argument("--worker-output",help=argparse.SUPPRESS);p.add_argument("--arm",choices=list("AB"),help=argparse.SUPPRESS)
    p.add_argument("--slot",type=int,choices=range(4),help=argparse.SUPPRESS)
    args=p.parse_args();protocol=read(resolve(args.protocol))
    helper,original,cohort=validate(protocol,resolve(args.candidate_milestone))
    if args.worker_output:return worker(args,protocol,helper,original,cohort)
    if not args.confirm_isolated:p.error("Wait for all other native workloads; then pass --confirm-isolated")
    if args.attempt!=protocol["attempt"]:p.error("This frozen protocol permits only --attempt prefix32-confirmation-1")
    out=ROOT/"evidence/05g-integration"/args.attempt;out.mkdir(parents=True,exist_ok=False)
    record={"status":"starting","script_sha256":sha(__file__),"protocol_sha256":sha(resolve(args.protocol)),
            "snapshots":snapshots(out,resolve(args.protocol)),"protocol":protocol,"cohort":cohort,
            "workers":[],"timings":[],"score_comparisons":[],"host_before":helper.host(),
            "isolation":"Operator asserted no concurrent native model/GPU/benchmark jobs; one resident full pipeline in each sequential worker.",
            "timing_boundary":"Synchronous Pipeline.forward only; fresh measured32+3 after excluded warmup32+3. Load/verification, state creation/disposal, score checks/saving and telemetry excluded.",
            "retry_policy":protocol["retry_policy"]}
    save(out/"results.json",record)
    try:
        import numpy as np
        expected=None;host_samples=[]
        for slot,arm in enumerate("ABBA"):
            output=out/f"slot{slot}-{arm}"
            command=[sys.executable,str(Path(__file__).resolve()),"--protocol",str(resolve(args.protocol)),
                     "--candidate-milestone",str(resolve(args.candidate_milestone)),"--worker-output",str(output),"--arm",arm,"--slot",str(slot)]
            print(f"starting P32 confirmation slot {slot} arm {arm}",flush=True)
            with (out/f"slot{slot}-{arm}.log").open("wb") as log:process=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            if process.returncode:raise RuntimeError(f"Worker {slot}/{arm} exited {process.returncode}; confirmation stops without retry")
            r=read(output/"results.json")
            if r.get("passed") is not True or r.get("status")!="completed" or r.get("cohort")!=cohort:raise ValueError("Worker evidence failed or cohort differs")
            host_samples.extend([r["host_before_timing"],r["host_after_timing"]])
            record["host_stability"]=host_stability(host_samples)
            if not record["host_stability"]["passed"]:raise ValueError("Observed power/thermal instability across workers; no promotion or retry")
            if sha(output/"scores.npz")!=r["scores_sha256"]:raise ValueError("Worker score archive identity differs")
            with np.load(output/"scores.npz") as data:scores={key:data[key].copy() for key in data.files}
            if set(scores)!={f"p32_s{i}" for i in range(4)}:raise ValueError("Worker score sites differ")
            if expected is None:expected=scores
            exact={key:bitwise_equal(scores[key],expected[key]) for key in sorted(scores)}
            record["score_comparisons"].append({"slot":slot,"arm":arm,"bitwise_exact":exact})
            if not all(exact.values()):raise ValueError("A/B score bytes differ; no promotion or retry")
            record["workers"].append({"slot":slot,"arm":arm,"result":str(output/"results.json"),"result_sha256":sha(output/"results.json"),
                                      "scores_sha256":r["scores_sha256"],"exit_code":process.returncode})
            record["timings"].extend(r["timings"]);save(out/"results.json",record)
        record["performance"]=summarize(record["timings"],protocol);record["host_after"]=helper.host()
        if record["host_stability"]["sample_count"]!=8 or not record["host_stability"]["passed"]:raise ValueError("All eight stable worker timing boundaries are required")
        if validate(protocol,resolve(args.candidate_milestone))[2]!=cohort or sha(resolve(args.protocol))!=record["protocol_sha256"]:raise ValueError("Frozen inputs changed during confirmation")
        go=record["performance"]["performance_gate_passed"]
        record.update(status="completed",experiment_completed=True,all_scores_bitwise_exact=True,
                      confirmation_pass=go,decision="P32 confirmation go; original complete qualification and benchmark remain required" if go else "confirmation no-go; candidate remains qualified but is not promoted; no further retry this turn")
    except BaseException as exc:
        record.update(status="failed",experiment_completed=False,confirmation_pass=False,
                      decision="candidate remains qualified but is not promoted; no further retry this turn",error={"type":type(exc).__name__,"message":str(exc)});raise
    finally:save(out/"results.json",record)
    print(json.dumps({"status":record["status"],"confirmation_pass":record["confirmation_pass"],"decision":record["decision"],"performance":record["performance"]},indent=2))
    return 0


if __name__=="__main__":raise SystemExit(main())
