"""Core ML CPU q·k tests with source-style layout and operator variants."""
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import coremltools as ct
import numpy as np
import torch
from torch import nn
from coreml_port.cleanup_guard import install
from portable.model import load_model
install();torch.set_num_threads(4)
OUT=ROOT/'evidence/04d-gpu/tiny3-dot-layouts';ART=ROOT/'artifacts/04d-gpu/tiny3-dot-layouts'
OUT.mkdir(parents=True,exist_ok=False);ART.mkdir(parents=True,exist_ok=False)
ids=[87555,130664,114854];m=load_model('cpu');attn=m.backbone.layers[32].self_attn;saved={};old=attn.project
def project(hidden,scale,ve,cos,sin):
    result=old(hidden,scale,ve,cos,sin);saved['q']=result[0].detach().clone();saved['k']=result[1].detach().clone();return result
attn.project=project
try:
    with torch.inference_mode():m.reference_hidden(torch.tensor([ids],dtype=torch.long))
finally:attn.project=old
q=saved['q'].float();k=saved['k'].repeat_interleave(5,dim=2).float()
native=torch.einsum('bthd,bshd->bhts',q,k)

class RawEinsum(nn.Module):
    def forward(self,q,k):return torch.einsum('bthd,bshd->bhts',q,k)
class RawMatmul(nn.Module):
    def forward(self,q,k):return q.permute(0,2,1,3)@k.permute(0,2,3,1)
class PerHead(nn.Module):
    def forward(self,q,k):
        return torch.stack([q[0,:,h,:]@k[0,:,h,:].T for h in range(10)],dim=0).unsqueeze(0)
class Reversed(nn.Module):
    def forward(self,q,k):
        return (k.permute(0,2,1,3)@q.permute(0,2,3,1)).transpose(-2,-1)

def check(actual,reference):
    diff=np.abs(actual.astype(np.float64)-reference.astype(np.float64));ix=np.argwhere(diff!=0)
    return {'different_elements':int(len(ix)),'max_abs':float(diff.max()),
            'first_differences':[{'index':i.tolist(),'actual':float(actual[tuple(i)]),'reference':float(reference[tuple(i)])} for i in ix[:8]]}
report={'ids':ids,'compute_units':'CPU_ONLY','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'operand_shape':{'q':list(q.shape),'k':list(k.shape)},'operand_stride':{'q':list(q.stride()),'k':list(k.stride())},
        'experiments':[]}
def save():(OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
save()
for name,module in [('raw_einsum',RawEinsum()),('raw_matmul',RawMatmul()),('per_head',PerHead()),('reversed',Reversed())]:
    module.eval()
    with torch.inference_mode():torch_dot=module(q,k);torch_scores=torch_dot*.1
    traced=torch.jit.trace(module,(q,k),check_trace=False)
    converted=ct.convert(traced,inputs=[ct.TensorType(name='q',shape=q.shape,dtype=np.float32),
                                        ct.TensorType(name='k',shape=k.shape,dtype=np.float32)],
                         outputs=[ct.TensorType(name='dot',dtype=np.float32)],
                         compute_precision=ct.precision.FLOAT32,minimum_deployment_target=ct.target.macOS15,skip_model_load=True)
    package=ART/(name+'.mlpackage');converted.save(str(package))
    cm=ct.models.MLModel(str(package),compute_units=ct.ComputeUnit.CPU_ONLY)
    actual=cm.predict({'q':q.numpy(),'k':k.numpy()})['dot']
    row={'name':name,'torch_vs_native_dot':check(torch_dot.numpy(),native.numpy()),
         'coreml_vs_torch_dot':check(actual,torch_dot.numpy()),
         'coreml_vs_native_dot':check(actual,native.numpy()),
         'coreml_vs_native_scores':check(np.float32(actual*np.float32(.1)),(native*.1).numpy()),
         'package_files_sha256':{str(p.relative_to(package)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(package.rglob('*')) if p.is_file()}}
    report['experiments'].append(row);print(name,{n:row[n]['different_elements'] for n in row if n.startswith(('torch_','coreml_'))},flush=True);save()
