"""Small direct numerical probes of the two pinned upstream implementations."""
import importlib.util
import json
from pathlib import Path
import sys
import pytest
import torch

from components import ROOT, configuration, randomize, upstream

runtime = ROOT / "sources/runtime/paradigma-inc-limite-violetto-cd68d27"
sys.path.insert(0, str(runtime / "src"))
spec = importlib.util.spec_from_file_location("contracts", runtime / "tests/test_package_contract.py")
contracts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contracts)


def difference(a, b):
    delta=(a.float()-b.float()).abs()
    return {"equal":bool(torch.equal(a,b)),"max_abs":float(delta.max()),"differing_elements":int(torch.count_nonzero(delta)),"elements":a.numel()}


with pytest.MonkeyPatch.context() as patch, torch.inference_mode():
    contracts._install_vllm_import_stubs(patch)
    from limite_vllm._components import LimiteMuddMixer, LimiteRotary, _AttentionBase
    config=configuration()
    gen=torch.Generator().manual_seed(20260922)
    value=upstream.GMPTMudd(config);randomize(value,20260923)
    serving=LimiteMuddMixer(config)
    serving.load_state_dict(value.state_dict(),strict=True)
    taps=[torch.randn(1,7,32,generator=gen).bfloat16() for _ in range(3)]
    current=torch.randn(1,7,32,generator=gen).bfloat16()
    mixers=[]
    for layer in (24,47):
        for r in (False,True):
            mixers.append({"layer":layer,"r_way":r,**difference(value(taps,current,layer,r_way=r),serving.combine(taps,current,layer,r_way=r))})
    attention=upstream.GMPTAttention(config,1)
    w=torch.randn(32,32,generator=gen).bfloat16();scale=torch.tensor(1.031)
    scalar=difference(attention._scaled(w,scale,torch.bfloat16),_AttentionBase._fold(scale,w))
    wrong_storage=difference(attention._scaled(w.float(),scale,torch.bfloat16),_AttentionBase._fold(scale,w))
    positions=torch.tensor([[0,1,4,1024,1025]])
    cos,sin=upstream.build_rope_factors(config,positions)
    q=torch.randn(1,5,4,8,generator=gen).bfloat16();k=torch.randn(1,5,2,8,generator=gen).bfloat16()
    rq,rk=LimiteRotary(config)(q[0],k[0],positions[0])
    rotary={"q":difference(upstream.apply_rotary(q,cos,sin)[0],rq),"k":difference(upstream.apply_rotary(k,cos,sin)[0],rk)}
    alpha=torch.randn(10,generator=gen)*.2
    precision=difference(alpha.tanh(),alpha.bfloat16().float().tanh())
    result={"kind":"upstream component probes on synthetic data; no checkpoint/model outputs",
            "mixed_precision_mudd_comparisons":mixers,"bf16_weight_scale_fold":scalar,
            "keeping_projection_weight_fp32_changes_fold":wrong_storage,
            "rotary":rotary,"rounding_xsa_alpha_before_fp32_tanh_changes_result":precision,
            "notes":["Value backbone _scaled docstring says FP32 multiply, but with BF16 weight and 0-dim FP32 scalar PyTorch uses BF16 promotion; observed equal to serving fold.","Do not promote checkpoint projection storage to FP32 before folding if trying to preserve the BF16 fold.","Mixer einsum and serving linear preserve identical BF16 output in these probes; this is not an all-input proof."]}
    target=ROOT/'evidence/02-components/upstream-qualification.json'
    target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
