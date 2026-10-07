"""Composed one-layer CPU Core ML test with runtime Q rows and raw einsum."""
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import coremltools as ct
import numpy as np
import torch
from torch import nn
import coreml_port.graphs as graphs
import coreml_port.ordered_attention as attention
from coreml_port.cleanup_guard import install
from coreml_port.ordered_linear import linear
from coreml_port.ordered_norm import NormWrapped
from coreml_port.ordered_xsa import ordered_xsa_dot,ordered_xsa_normalize
from coreml_port.source_exp import exp
from portable.model import load_model
from scripts.gpu_dynamic_raw_attention import finish_parts
install();torch.set_num_threads(4)
graphs.linear=linear;graphs.softmax=attention.softmax;attention.exponential=exp
graphs.xsa_normalize=ordered_xsa_normalize;graphs.xsa_dot=ordered_xsa_dot
OUT=ROOT/'evidence/04d-gpu/tiny3-dynamic-layer32';ART=ROOT/'artifacts/04d-gpu/tiny3-dynamic-layer32'
OUT.mkdir(parents=True,exist_ok=False);ART.mkdir(parents=True,exist_ok=False)
PACKAGE=ART/'layer32.mlpackage';ids=[87555,130664,114854]
report={'ids':ids,'compute_units':'CPU_ONLY','row_policy':'front-pad; min(32,last_absolute_position+1)',
        'key_capacity':2048,'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            ROOT/'scripts/gpu_tiny3_dynamic_layer32.py',ROOT/'scripts/gpu_dynamic_raw_attention.py',
            ROOT/'coreml_port/graphs.py',ROOT/'coreml_port/ordered_attention.py',ROOT/'coreml_port/source_exp.py',
            ROOT/'coreml_port/ordered_norm.py',ROOT/'coreml_port/ordered_linear.py',ROOT/'coreml_port/ordered_xsa.py')},
        'status':'starting'}
def save():(OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
def compare(actual,expected):
    a=actual.astype(np.float32) if isinstance(actual,np.ndarray) else actual.detach().float().numpy()
    b=expected.detach().float().numpy();diff=np.abs(a.astype(np.float64)-b.astype(np.float64));ix=np.argwhere(diff!=0)
    return {'different_elements':int(len(ix)),'max_abs':float(diff.max()),
            'first_differences':[{'index':i.tolist(),'actual':float(a[tuple(i)]),'expected':float(b[tuple(i)])} for i in ix[:8]]}
save();model=load_model('cpu');attn=model.backbone.layers[32].self_attn;layer=model.backbone.layers[32]
ref={};old_project=attn.project;old_eager=attn._eager
def project(hidden,scale,ve,cos,sin):
    result=old_project(hidden,scale,ve,cos,sin)
    ref.update(q=result[0].detach().clone(),k=result[1].detach().clone(),v=result[2].detach().clone())
    return result
def eager(q,k,v,seq_ids):
    out=old_eager(q,k,v,seq_ids);ref['weighted']=out.detach().clone();return out
def pre(module,args):
    ref['input']=args[1].detach().clone();ref['residual']=args[2].detach().clone()
def post(module,args,output):ref['layer_output']=output.detach().clone()
attn.project=project;attn._eager=eager;hp=layer.register_forward_pre_hook(pre);ho=layer.register_forward_hook(post)
try:
    with torch.inference_mode():model.reference_hidden(torch.tensor([ids],dtype=torch.long))
finally:
    attn.project=old_project;attn._eager=old_eager;hp.remove();ho.remove()

class Standalone(nn.Module):
    def __init__(self,model):
        super().__init__();self.layer=graphs.Layer(model,32).eval()
        self.register_buffer('slots',torch.arange(2048,dtype=torch.int32))
    def forward(self,x,residual,q,keys,values,v,positions):
        return finish_parts(self.layer,x,residual,q,keys,values,v,positions,self.slots)

keys=torch.zeros(1,2048,2,128,dtype=torch.float32);values=torch.zeros_like(keys)
keys[:,:3]=ref['k'].to(torch.float16).float();values[:,:3]=ref['v'].to(torch.float16).float()
args=(ref['input'].float(),ref['residual'].float(),ref['q'].float(),keys,values,ref['v'].float(),
      torch.arange(3,dtype=torch.int32),torch.tensor([1280.]))
graph=NormWrapped(Standalone(model).eval()).eval()
with torch.inference_mode(): torch_y,torch_weighted=graph(*args)
report['torch_y']=compare(torch_y,ref['layer_output']);report['torch_weighted']=compare(torch_weighted,ref['weighted']);save()
trace=torch.jit.trace(graph,args,check_trace=False)
input_names=('x','residual','q','keys','values','v','positions','norm_divisor')
inputs=[ct.TensorType(name=n,shape=t.shape,dtype=np.int32 if n=='positions' else np.float32) for n,t in zip(input_names,args)]
converted=ct.convert(trace,inputs=inputs,outputs=[ct.TensorType(name='y',dtype=np.float32),
                                                   ct.TensorType(name='weighted',dtype=np.float32)],
    compute_precision=ct.precision.FLOAT32,minimum_deployment_target=ct.target.macOS15,skip_model_load=True)
converted.save(str(PACKAGE));report['status']='converted'
report['package_files_sha256']={str(p.relative_to(PACKAGE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(PACKAGE.rglob('*')) if p.is_file()};save()
cm=ct.models.MLModel(str(PACKAGE),compute_units=ct.ComputeUnit.CPU_ONLY)
got=cm.predict({n:t.numpy() for n,t in zip(input_names,args)})
report['coreml_y']=compare(got['y'],ref['layer_output']);report['coreml_weighted']=compare(got['weighted'],ref['weighted'])
report['coreml_vs_torch_y']=compare(got['y'],torch_y);report['coreml_vs_torch_weighted']=compare(got['weighted'],torch_weighted)
report['status']='complete';save();print(report['coreml_weighted'],report['coreml_y'],flush=True)
