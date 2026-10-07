"""Compare BF16 projection kernels with wider and compensated reductions."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
import torch.nn.functional as F
from portable.model import load_model,rms
from coreml_port.graphs import Q

torch.set_num_threads(4);m=load_model();ids=torch.tensor([json.loads((ROOT/'evidence/01-reference/local-input-validation.json').read_text())['cases'][0]['input_ids'][:32]])
def count(a,b):return int((a.float()!=b.float()).sum())
def blocked(x,w,size,compensate):
    total=torch.zeros(x.shape[:-1]+(w.shape[0],));correction=torch.zeros_like(total)
    for start in range(0,w.shape[1],size):
        part=F.linear(x[...,start:start+size],w[:,start:start+size])
        if compensate:
            y=part-correction;t=total+y;correction=(t-total)-y;total=t
        else:total=total+part
    return total.bfloat16()
r={}
with torch.inference_mode():
    captured={};m(ids,capture=lambda n,v:captured.update({n:v.clone()}))
    for i in [0,1,12,24,36,47]:
        layer=m.backbone.layers[i];x=captured[f'layer_{i}_mixer_attention'] if i in [24,47] else rms(captured['embedding_norm' if i==0 else f'layer_{i-1}'])
        for name in ['q','k','v']:
            w=layer.self_attn._scaled(getattr(layer.self_attn,name+'_proj').weight,layer.scales.qkv,torch.bfloat16);want=F.linear(x,w);xf=x.float();wf=w.float();fp=F.linear(xf,wf);exact=F.linear(x.double(),w.double())
            v={'elements':want.numel(),'fp32':count(fp.bfloat16(),want),'fp64_direct':count(exact.bfloat16(),want),'fp64_via_fp32':count(exact.float().bfloat16(),want),'rounding_formula_vs_native':count(Q(fp),fp.bfloat16())}
            for size in [32,64,128,256]:
                for comp in [False,True]:v[f'block{size}_kahan{comp}']=count(blocked(xf,wf,size,comp),want)
            r[f'{i}/{name}']=v;print(i,name,v,flush=True)
(ROOT/'evidence/04c-precision/projection-accumulation.json').write_text(json.dumps(r,indent=2)+'\n')
