"""Locate backend-dependent differences without changing acceptance limits."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from portable.model import load_model,upstream,rms,attention_mask
from probe_full_model import metrics

torch.set_num_threads(4)
m=load_model('mps');ids=json.loads((ROOT/'evidence/01-reference/local-input-validation.json').read_text())['cases'][0]['input_ids']
x=torch.tensor([ids],device=m.device);n=x.shape[1]
positions=torch.arange(n,device=m.device);cos,sin=upstream.build_rope_factors(m.config,positions[None])
h=rms(m.backbone.embed_tokens(x));ve=m.backbone._value_embeddings(x)
layer=m.backbone.layers[0];attn=layer.self_attn
q,k,v=attn.project(rms(h),layer.scales.qkv,ve,cos,sin)
def attention(q,k,v,pos):
    k=k.repeat_interleave(5,dim=2);v=v.repeat_interleave(5,dim=2)
    weights=torch.einsum('bthd,bshd->bhts',q.float(),k.float())*.1
    mask=attention_mask(pos,k.shape[1],1024)
    weights=weights.masked_fill(~mask[None,None],float('-inf')).softmax(-1)
    return torch.einsum('bhts,bshd->bthd',weights,v.float())
full=attention(q,k,v,positions)
result=[]
for i in [41,42,43]:
    qi,ki,vi=attn.project(rms(h[:,i:i+1]),layer.scales.qkv,ve[:,i:i+1],cos[:,i:i+1],sin[:,i:i+1])
    a=attention(q[:,i:i+1],k,v,positions[i:i+1])
    b=attention(q[:,i:i+1],k[:,:i+1],v[:,:i+1],positions[i:i+1])
    entry={'position':i,'q_projection':metrics(q[:,i:i+1],qi),'k_projection':metrics(k[:,i:i+1],ki),'v_projection':metrics(v[:,i:i+1],vi),'one_query_same_keys_fp32':metrics(full[:,i:i+1],a),'one_query_short_keys_fp32':metrics(full[:,i:i+1],b),'one_query_same_keys_bf16':metrics(full[:,i:i+1].bfloat16(),a.bfloat16()),'one_query_short_keys_bf16':metrics(full[:,i:i+1].bfloat16(),b.bfloat16())}
    result.append(entry)
path=ROOT/'evidence/03-portable/mps-first-layer-diagnosis.json'
path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
