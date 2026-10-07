"""Isolate Core ML execution from graph-level linear and RMS arithmetic."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np,coremltools as ct
from torch import nn
from portable.model import load_model,rms
from coreml_port.graphs import Q,norm
from coreml_port.ordered_linear import linear
from coreml_port.cleanup_guard import install
from scripts.probe_full_model import metrics
install();torch.set_num_threads(4);m=load_model();ids=torch.tensor([json.loads((ROOT/'evidence/01-reference/local-input-validation.json').read_text())['cases'][0]['input_ids'][:32]])
with torch.inference_mode():
    x=rms(rms(m.backbone.embed_tokens(ids))).float();l=m.backbone.layers[0];w=l.self_attn._scaled(l.self_attn.q_proj.weight,l.scales.qkv,torch.bfloat16).float()
class Operators(nn.Module):
    def __init__(self):super().__init__();self.register_buffer('w',w)
    def forward(self,x):
        raw=linear(x,self.w)
        return raw,Q(raw),norm(x),Q(x),x*torch.rsqrt((x*x).mean(-1,keepdim=True)+torch.finfo(torch.float32).eps)
g=Operators().eval();trace=torch.jit.trace(g,x[:,:3],check_trace=False);names=['raw','rounded','norm','identity_round','norm_raw'];path=ROOT/'artifacts/04c-precision/operators.mlpackage'
converted=ct.convert(trace,inputs=[ct.TensorType(name='x',shape=(1,ct.RangeDim(1,32,default=3),1280),dtype=np.float32)],outputs=[ct.TensorType(name=n,dtype=np.float32) for n in names],compute_precision=ct.precision.FLOAT32,minimum_deployment_target=ct.target.macOS15,skip_model_load=True);converted.save(str(path));r={};captures={}
for units in [ct.ComputeUnit.CPU_ONLY,ct.ComputeUnit.CPU_AND_GPU]:
    cm=ct.models.MLModel(str(path),compute_units=units);r[units.name]={}
    with torch.inference_mode():
        for count in [32,1,3]:
            v=x[:,:count];want=g(v);got=cm.predict({'x':v.numpy()});result={}
            for n,e in zip(names,want):
                a=got[n];result[n]={**metrics(torch.from_numpy(a),e),'different_elements':int(np.count_nonzero(a!=e.numpy()))};captures[f'{units.name}_{count}_{n}_got']=a;captures[f'{units.name}_{count}_{n}_want']=e.numpy()
            r[units.name][str(count)]=result;print(units.name,count,result,flush=True)
(ROOT/'evidence/04c-precision/operators.json').write_text(json.dumps(r,indent=2)+'\n');np.savez_compressed(ROOT/'artifacts/04c-precision/operators-captures.npz',**captures)
