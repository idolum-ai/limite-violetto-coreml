"""Localize graph-candidate precision drift before Core ML optimization."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
import torch.nn.functional as F
from portable.model import load_model,rms,upstream
from coreml_port.graphs import Q,norm,Layer
from scripts.probe_full_model import metrics

torch.set_num_threads(4);m=load_model();ids=torch.tensor([json.loads((ROOT/'evidence/01-reference/local-input-validation.json').read_text())['cases'][0]['input_ids'][:32]])
r={}
with torch.inference_mode():
    emb=m.backbone.embed_tokens(ids);x=rms(emb);xf=norm(emb.float());r['embedding_norm']=metrics(xf,x)
    for i,source in enumerate(m.backbone.layers):
        # Feed identical real source activations into individual operations.
        captures={};break
    m(ids,capture=lambda n,v:captures.update({n:v.clone()}))
    for i,source in enumerate(m.backbone.layers):
        h=captures['embedding_norm' if i==0 else f'layer_{i-1}']
        if i in [24,47]:h=captures[f'layer_{i}_mixer_attention'];xf=h.float()
        else:xf=norm(h.float());h=rms(h)
        a=source.self_attn;g=Layer(m,i);entry={'input_norm':metrics(xf,h)}
        for n in ['q','k','v']:
            w=a._scaled(getattr(a,n+'_proj').weight,source.scales.qkv,torch.bfloat16)
            want=F.linear(h,w);got=Q(F.linear(h.float(),w.float()))
            entry[n+'_projection']={**metrics(got,want),'different_elements':int((got!=want.float()).sum()),'elements':want.numel()}
        cos,sin=upstream.build_rope_factors(m.config,torch.arange(32)[None,:]);q,k,v=a.project(h,source.scales.qkv,m.backbone._value_embeddings(ids),cos,sin)
        entry['kv_fp16']={'k_max':float(k.abs().max()),'v_max':float(v.abs().max()),'k_changed':int((k.float()!=k.half().float()).sum()),'v_changed':int((v.float()!=v.half().float()).sum()),'finite':bool(k.half().isfinite().all() and v.half().isfinite().all())}
        r[str(i)]=entry
(ROOT/'evidence/04-coreml/precision-diagnosis.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'layers_with_projection_differences':[i for i in map(str,range(48)) if any(r[i][n+'_projection']['different_elements'] for n in ['q','k','v'])],'layers_with_cache_narrowing':[i for i in map(str,range(48)) if r[i]['kv_fp16']['k_changed'] or r[i]['kv_fp16']['v_changed']]},indent=2))
