"""Isolate CPU Core ML attention differences with identical captured inputs.

Diagnostic outputs may change fusion. These tests do not qualify the runner.
Every invocation uses a new run directory and records its script hash.
"""
import argparse, hashlib, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch import nn
import coremltools as ct
import coreml_port.graphs as graphs
from coreml_port.ordered_linear import linear
from coreml_port.cleanup_guard import install
from portable.model import load_model
from scripts.probe_full_model import metrics

p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
out=ROOT/'evidence/04d-cpu'/a.run;art=ROOT/'artifacts/04d-cpu'/a.run
out.mkdir(parents=True,exist_ok=False);art.mkdir(parents=True,exist_ok=False)
install();torch.set_num_threads(4);graphs.linear=linear
capture=ROOT/'artifacts/04d-cpu/attention-inputs.npz'
if not capture.exists():
    m=load_model();ids=torch.tensor([json.loads((ROOT/'evidence/01-reference/local-input-validation.json').read_text())['cases'][0]['input_ids'][:32]])
    g=graphs.Group(m,0).eval();saved={};original=graphs.Layer.finish
    def intercept(self,x,residual,q,keys,values,v,positions,slots):
        if self.i==1:
            saved.update(q=q.permute(0,2,1,3).clone(),k=keys.repeat_interleave(5,dim=2).permute(0,2,3,1).clone(),v=values.repeat_interleave(5,dim=2).permute(0,2,1,3).clone(),mask=(slots[None,:]<=positions[:,None])[None,None])
        return original(self,x,residual,q,keys,values,v,positions,slots)
    graphs.Layer.finish=intercept
    try:
        with torch.inference_mode():
            x,ve=graphs.Embedding(m)(ids);g(x,ve,torch.arange(32,dtype=torch.int32),x,x,x)
    finally:graphs.Layer.finish=original
    np.savez(capture,**{k:v.numpy() for k,v in saved.items()})
    del g,m
data=np.load(capture);q,k,v,mask=[torch.from_numpy(data[n]) for n in ['q','k','v','mask']]
class Attention(nn.Module):
    def forward(self,q,k,v,mask):
        dot=q@k;scores=dot*.1;weights=scores.masked_fill(mask<.5,float('-inf')).softmax(-1);raw=weights@v
        return dot,scores,weights,raw,graphs.Q(raw)
class Softmax(nn.Module):
    def forward(self,scores,mask):return scores.masked_fill(mask<.5,float('-inf')).softmax(-1)
class Product(nn.Module):
    def forward(self,weights,v):
        y=weights@v
        return y,graphs.Q(y)
class Decomposed(nn.Module):
    def forward(self,scores,mask,v):
        shifted=scores.masked_fill(mask<.5,float('-inf'))
        shifted=shifted-shifted.max(-1,keepdim=True).values
        ex=shifted.exp();total=ex[...,:4]
        for j in range(4,ex.shape[-1],4):total=total+ex[...,j:j+4]
        total=(total[...,0:1]+total[...,2:3])+(total[...,1:2]+total[...,3:4])
        inv=torch.ones_like(total)/total;w=ex*inv;raw=w@v
        return shifted,ex,total,inv,w,raw,graphs.Q(raw)

report={'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'capture_sha256':hashlib.sha256(capture.read_bytes()).hexdigest(),'compute':'CPU_ONLY','diagnostic_only':True,'experiments':{}}
def run(name,module,args,names,outputs):
    with torch.inference_mode():expected=module(*args)
    if isinstance(expected,torch.Tensor):expected=(expected,)
    trace=torch.jit.trace(module,args,check_trace=False)
    model=ct.convert(trace,inputs=[ct.TensorType(name=n,shape=t.shape,dtype=np.float32) for n,t in zip(names,args)],outputs=[ct.TensorType(name=n,dtype=np.float32) for n in outputs],compute_precision=ct.precision.FLOAT32,minimum_deployment_target=ct.target.macOS15,skip_model_load=True)
    path=art/(name+'.mlpackage');model.save(str(path));cm=ct.models.MLModel(str(path),compute_units=ct.ComputeUnit.CPU_ONLY)
    got=cm.predict({n:t.numpy() for n,t in zip(names,args)});result=[];arrays={}
    for name,key,want in zip(outputs,[x.name for x in cm.get_spec().description.output],expected):
        actual=got[key];target=want.numpy();diff=np.argwhere(actual!=target)
        r={'name':name,**metrics(torch.from_numpy(actual),want),'different_elements':len(diff),'first_differences':[{'index':ix.tolist(),'actual':float(actual[tuple(ix)]),'expected':float(target[tuple(ix)])} for ix in diff[:8]]};result.append(r)
        arrays[name+'_actual']=actual;arrays[name+'_expected']=target
        print(r,flush=True)
    np.savez_compressed(out/(path.stem+'-outputs.npz'),**arrays)
    report['experiments'][path.stem]=result;(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    return expected
with torch.inference_mode():
    dot,scores,weights,raw,rounded=run('attention',Attention(),(q,k,v,mask.float()),['q','k','v','mask'],['dot','scores','weights','raw','rounded'])
    run('softmax',Softmax(),(scores,mask.float()),['scores','mask'],['weights'])
    run('product',Product(),(weights,v),['weights','v'],['raw','rounded'])
    run('decomposed',Decomposed(),(scores[...,:32],mask.float()[...,:32],v[...,:32,:]),['scores','mask','v'],['shifted','exp','sum','reciprocal','weights','raw','rounded'])
