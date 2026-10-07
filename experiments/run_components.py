"""Run one bounded component experiment and retain measurements/artifacts."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import time
import traceback

import coremltools as ct
import numpy as np
import torch

from components import ROOT, Attention, Mixer

EVIDENCE = ROOT / "evidence/02-components"
ACCEPTANCE = json.loads((EVIDENCE / "ACCEPTANCE.json").read_text())


def array(t):
    return t.detach().float().numpy()


def metrics(actual, expected, rule):
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    if a.shape != b.shape:
        return {"passed": False, "actual_shape": list(a.shape), "expected_shape": list(b.shape)}
    d = np.abs(a - b)
    output = {"shape": list(a.shape), "finite": bool(np.isfinite(a).all() and np.isfinite(b).all()),
              "max_abs": float(d.max()), "mean_abs": float(d.mean()),
              "max_relative_floor_1e-6": float((d / np.maximum(np.abs(b), 1e-6)).max())}
    if rule:
        limits = ACCEPTANCE["comparisons"][rule]
        output["passed"] = output["finite"] and bool(np.all(d <= limits["atol"] + limits["rtol"] * np.abs(b)))
    return output


def convert(module, example, inputs, outputs, directory, states=None):
    start = time.perf_counter()
    traced = torch.jit.trace(module, example, check_trace=False)
    if hasattr(module, "reset"):
        module.reset()
    traced.save(str(directory / "traced.pt"))
    converted = ct.convert(traced, inputs=inputs, outputs=outputs, states=states,
                           convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
                           compute_precision=ct.precision.FLOAT32, skip_model_load=True)
    (directory / "mil.txt").write_text(str(converted._mil_program))
    package = directory / "model.mlpackage"
    converted.save(str(package))
    convert_seconds = time.perf_counter() - start
    start = time.perf_counter()
    model = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_AND_GPU)
    return model, {"convert_seconds": convert_seconds, "compile_load_seconds": time.perf_counter() - start,
                   "compute_units_allowed": "CPU_AND_GPU", "actual_placement": "not profiled",
                   "compute_precision": "FLOAT32; FP16 states", "package": str(package.relative_to(ROOT))}


def attention(args, directory):
    module = Attention(args.production, args.global_attention)
    g = torch.Generator().manual_seed(20260922)
    steps = ([32, 1, 7, 216, 256, 256, 256, 1, 1, 1, 1, 1, 1, 1] if args.production
             else [3, 1, 2, 1, 1, 4])
    total = sum(steps)
    x = torch.randn(1, total, module.hidden, generator=g) * 0.25
    ve = torch.randn(1, total, module.kv, module.d, generator=g) * 0.25
    initial_n = min(17, total)
    local = module.full_prefix(x[:, :initial_n], ve[:, :initial_n])
    upstream = module.upstream_reference(x[:, :initial_n], ve[:, :initial_n])
    bf16 = module.upstream_reference(x[:, :initial_n], ve[:, :initial_n], bf16=True)
    checks = {"rewrite_vs_upstream_fp32": metrics(array(local), array(upstream), "portable_fp32_rewrite_vs_upstream_fp32"),
              "candidate_vs_upstream_bf16_diagnostic": metrics(array(local), array(bf16), None)}
    full = array(module.full_prefix(x, ve, round_cache=True))
    module.reset()
    q = ct.RangeDim(1, 256 if args.production else 16, default=3)
    model, timing = convert(module, (x[:, :3], ve[:, :3], torch.arange(3, dtype=torch.int32)),
        [ct.TensorType(name="x", shape=(1, q, module.hidden), dtype=np.float32),
         ct.TensorType(name="ve", shape=(1, q, module.kv, module.d), dtype=np.float32),
         ct.TensorType(name="positions", shape=(q,), dtype=np.int32)],
        [ct.TensorType(name="y", dtype=np.float32)], directory,
        states=[ct.StateType(name=n, wrapped_type=ct.TensorType(shape=tuple(module.k_cache.shape), dtype=np.float16))
                for n in ("k_cache", "v_cache")])
    module.reset()
    state = model.make_state()
    results, torch_results, calls, comparisons = [], [], [], []
    start_pos = 0

    def inputs(start, count, xx=x, vv=ve):
        return {"x": array(xx[:, start:start+count]), "ve": array(vv[:, start:start+count]),
                "positions": np.arange(start, start+count, dtype=np.int32)}

    for count in steps:
        pos = torch.arange(start_pos, start_pos + count, dtype=torch.int32)
        expected = array(module(x[:, start_pos:start_pos+count], ve[:, start_pos:start_pos+count], pos))
        feed = inputs(start_pos, count)
        start = time.perf_counter()
        actual = model.predict(feed, state=state)["y"]
        elapsed = time.perf_counter() - start
        comparison = {"start": start_pos, "count": count,
            "output": metrics(actual, expected, "coreml_vs_same_precision_torch"),
            "cached_vs_full": metrics(expected, full[:, start_pos:start_pos+count], "chunk_vs_full_with_same_cache_rounding")}
        for name in ("k_cache", "v_cache"):
            comparison[name] = metrics(state.read_state(name), getattr(module, name).numpy(), "cache_state_vs_torch")
        comparisons.append(comparison)
        calls.append({"start": start_pos, "count": count, "seconds": elapsed})
        results.append(actual)
        torch_results.append(expected)
        start_pos += count
    # Request isolation: interleave B with an already populated A state.
    saved_a = {name: state.read_state(name).copy() for name in ("k_cache", "v_cache")}
    b_input = inputs(0, 3, -x, ve * 0.75)
    b = model.make_state()
    for name in ("k_cache", "v_cache"):
        checks["fresh_state_zero_" + name] = metrics(b.read_state(name), np.zeros(tuple(module.k_cache.shape)), "fresh_request_and_masked_slot_invariance")
    b_out = model.predict(b_input, state=b)["y"]
    b_alone = model.predict(b_input, state=model.make_state())["y"]
    checks["fresh_request_B"] = metrics(b_out, b_alone, "fresh_request_and_masked_slot_invariance")
    checks["B_vs_clean_torch_prefix"] = metrics(b_out, array(module.full_prefix(-x[:, :3],ve[:, :3]*0.75,round_cache=True)), "coreml_vs_same_precision_torch")
    for name in saved_a:
        checks["A_unchanged_after_B_" + name] = metrics(state.read_state(name), saved_a[name], "fresh_request_and_masked_slot_invariance")
    # Poison future/unused slots; masks must remove their influence.
    poisoned = model.make_state()
    for name in ("k_cache", "v_cache"):
        # Native Python state writer accepts float32 arrays and narrows to the
        # declared FP16 state; its NumPy float16 input path rejects the value.
        poisoned.write_state(name, np.full(tuple(module.k_cache.shape), 32.0, dtype=np.float32))
    normal_input = inputs(0, 3)
    clean = model.predict(normal_input, state=model.make_state())["y"]
    poison_out = model.predict(normal_input, state=poisoned)["y"]
    checks["unused_cache_poison"] = metrics(poison_out, clean, "fresh_request_and_masked_slot_invariance")
    future = {name: value.copy() for name, value in normal_input.items()}
    future["x"][:, -1] *= -3
    future["ve"][:, -1] *= -3
    future_out = model.predict(future, state=model.make_state())["y"]
    checks["prefill_future_causality"] = metrics(future_out[:, :-1], clean[:, :-1], "fresh_request_and_masked_slot_invariance")
    # Semantic impulse: attention at q=1024 must include key 0; q=1025 excludes it.
    # The global graph must retain that key at both positions. State injection avoids
    # turning this boundary test into a test of a thousand projection operations.
    boundary = []
    if args.production:
        zero_x = np.zeros((1, 1, module.hidden), dtype=np.float32)
        zero_ve = np.zeros((1, 1, module.kv, module.d), dtype=np.float32)
        for pos in (1024, 1025):
            bs = model.make_state()
            values = np.zeros(tuple(module.v_cache.shape), dtype=np.float16)
            values[:, 0] = 64
            bs.write_state("v_cache", values.astype(np.float32))
            got = model.predict({"x":zero_x, "ve":zero_ve, "positions":np.array([pos],dtype=np.int32)}, state=bs)["y"]
            module.reset()
            module.v_cache.copy_(torch.from_numpy(values))
            want = array(module(torch.from_numpy(zero_x), torch.from_numpy(zero_ve), torch.tensor([pos],dtype=torch.int32)))
            expected_visible = args.global_attention or pos == 1024
            observed_visible = bool(np.max(np.abs(got)) > 1e-5)
            boundary.append({"query_position":pos,"key_zero_should_be_visible":expected_visible,
                             "observed_visible":observed_visible,"passed":expected_visible==observed_visible,
                             "comparison":metrics(got,want,"coreml_vs_same_precision_torch"),
                             "max_abs_output":float(np.max(np.abs(got)))})
    np.savez(directory / "captures.npz", coreml=np.concatenate(results,axis=1),
             torch=np.concatenate(torch_results,axis=1), full_prefix=full,
             upstream_fp32=array(upstream), upstream_bf16=array(bf16))
    decode_calls = [call["seconds"] for call in calls if call["count"] == 1]
    warm_decode = decode_calls[1:]  # Exclude first use of the single-token shape.
    return {"checks": checks, "updates": comparisons, "boundary": boundary, "timing": timing,
            "calls": calls, "warm_decode_median_seconds": statistics.median(warm_decode),
            "first_single_token_shape_seconds": decode_calls[0],
            "dimensions": {"hidden":module.hidden,"query_heads":module.h,"kv_heads":module.kv,
                           "head_dim":module.d,"capacity":module.capacity,"window_distance":module.window},
            "state_bytes": module.k_cache.numel() * 4,
            "total_tokens":total, "weight_source":"deterministic synthetic, not checkpoint"}


def mixer(args, directory):
    module = Mixer(args.production)
    h = module.config.hidden_size
    g = torch.Generator().manual_seed(20260922)
    taps = torch.randn(3, 1, 7, h, generator=g) * .25
    current = torch.randn(1, 7, h, generator=g) * .25
    ours, ref, bf16 = module(taps,current), module.upstream_reference(taps,current), module.upstream_reference(taps,current,bf16=True)
    checks = {}
    for i in range(4):
        checks[f"rewrite_vs_upstream_{i}"] = metrics(array(ours[i]),array(ref[i]),"portable_fp32_rewrite_vs_upstream_fp32")
        checks[f"bf16_diagnostic_{i}"] = metrics(array(ours[i]),array(bf16[i]),None)
    q = ct.RangeDim(1,32,default=7)
    model,timing = convert(module,(taps,current),
        [ct.TensorType(name="taps",shape=(3,1,q,h),dtype=np.float32),ct.TensorType(name="current",shape=(1,q,h),dtype=np.float32)],
        [ct.TensorType(name=f"y{i}",dtype=np.float32) for i in range(4)],directory)
    calls=[]
    captures={}
    for n in (1,7,17):
        t=torch.randn(3,1,n,h,generator=g)*.25;c=torch.randn(1,n,h,generator=g)*.25
        start=time.perf_counter();actual=model.predict({"taps":array(t),"current":array(c)});calls.append(time.perf_counter()-start)
        expected=module(t,c)
        for i in range(4):
            checks[f"coreml_n{n}_path{i}"]=metrics(actual[f"y{i}"],array(expected[i]),"coreml_vs_same_precision_torch")
            captures[f"coreml_n{n}_path{i}"]=actual[f"y{i}"]
            captures[f"torch_n{n}_path{i}"]=array(expected[i])
    np.savez(directory/"captures.npz",**captures)
    return {"checks":checks,"timing":timing,"call_seconds":calls,"hidden":h,"sites":[24,47],"paths":["H","R"]}


def failures(value, path=""):
    found=[]
    if isinstance(value,dict):
        if value.get("passed") is False:found.append(path)
        for k,v in value.items():found.extend(failures(v,path+"/"+k))
    elif isinstance(value,list):
        for i,v in enumerate(value):found.extend(failures(v,path+"/"+str(i)))
    return found


def main():
    p=argparse.ArgumentParser();p.add_argument("kind",choices=["attention","mixer"]);p.add_argument("--production",action="store_true");p.add_argument("--global-attention",action="store_true");p.add_argument("--attempt",default="1");args=p.parse_args()
    torch.set_num_threads(4)
    name=f"{args.kind}-{'production' if args.production else 'small'}"+("-global" if args.global_attention else "-local" if args.kind=="attention" else "")+f"-attempt{args.attempt}"
    directory=ROOT/"artifacts/02-components"/name;directory.mkdir(parents=True,exist_ok=False)
    result={"name":name,"python":platform.python_version(),"torch":torch.__version__,"coremltools":ct.__version__,
            "acceptance_sha256":hashlib.sha256((EVIDENCE/"ACCEPTANCE.json").read_bytes()).hexdigest(),
            "sources_sha256":{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),Path(__file__).with_name('components.py')]},
            "artifact_path":str(directory.relative_to(ROOT))}
    start=time.perf_counter()
    try:
        with torch.inference_mode():
            result.update(attention(args,directory) if args.kind=="attention" else mixer(args,directory))
        result["failed_checks"]=failures(result)
        result["status"]="PASS" if not result["failed_checks"] else "FAIL"
    except Exception:
        result["status"]="ERROR";result["traceback"]=traceback.format_exc();print(result["traceback"],flush=True)
    result["elapsed_seconds"]=time.perf_counter()-start
    result["process_peak_rss_gib"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30
    if "timing" in result:
        limits=ACCEPTANCE["resource_gate"]
        result["resource_pass"]=result["process_peak_rss_gib"]<limits["max_single_experiment_process_peak_rss_gib"] and result["timing"]["compile_load_seconds"]<limits["max_production_component_compile_seconds"] and result.get("warm_decode_median_seconds",0)<limits["max_warm_single_token_call_seconds"]
    (EVIDENCE/f"{name}.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in {"checks","updates","boundary","calls","traceback"}},indent=2),flush=True)
    if result["status"]!="PASS":raise SystemExit(1)


if __name__=="__main__":main()
