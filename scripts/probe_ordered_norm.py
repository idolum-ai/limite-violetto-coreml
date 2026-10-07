"""Source-derived RMS sum-order check on captured and independent inputs."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np,coremltools as ct
from torch import nn
from coreml_port import graphs
from coreml_port.ordered_norm import ordered_sum,raw_norm
from coreml_port.cleanup_guard import install
from portable.model import EPS
install();torch.set_num_threads(4)
out=ROOT/'evidence/04d-cpu/ordered-norm';art=ROOT/'artifacts/04d-cpu/ordered-norm';out.mkdir(parents=True,exist_ok=False);art.mkdir(parents=True,exist_ok=False)
x=torch.from_numpy(np.load(ROOT/'artifacts/04d-gpu/cpu-norm/norm-input.npz')['value'])
report={'source_sha256':hashlib.sha256((ROOT/'coreml_port/ordered_norm.py').read_bytes()).hexdigest(),'torch_checks':[],'coreml':{}}
generator=torch.Generator().manual_seed(20260922)
with torch.inference_mode():
    for name,v in [('capture',x),('random128',torch.randn(2,37,128,generator=generator).bfloat16().float()),('random1280',torch.randn(2,37,1280,generator=generator).bfloat16().float()),('scaled1280',torch.randn(2,37,1280,generator=generator).bfloat16().float()*4096)]:
        squared=v*v;exact=ordered_sum(squared);want=squared.sum(-1,keepdim=True)
        report['torch_checks'].append({'name':name,'sum_differences':int((exact!=want).sum()),'mean_differences':int((exact/v.shape[-1]!=squared.mean(-1,keepdim=True)).sum()),'rounded_norm_differences':int((graphs.Q(raw_norm(v,torch.tensor([float(v.shape[-1])])))!=graphs.norm(v)).sum())})
class Diagnostic(nn.Module):
    def forward(self,x,divisor):
        total=ordered_sum(x*x);mean=total/divisor;inv=torch.ones_like(mean)/torch.sqrt(mean+EPS);raw=x*inv
        return total,mean,inv,raw,graphs.Q(raw)
with torch.inference_mode():expected=Diagnostic()(x,torch.tensor([1280.]))
names=['sum','mean','inv','raw','rounded'];traced=torch.jit.trace(Diagnostic().eval(),(x,torch.tensor([1280.])),check_trace=False)
cm=ct.convert(traced,inputs=[ct.TensorType(name='x',shape=x.shape,dtype=np.float32),ct.TensorType(name='divisor',shape=(1,),dtype=np.float32)],outputs=[ct.TensorType(name=n,dtype=np.float32) for n in names],compute_precision=ct.precision.FLOAT32,minimum_deployment_target=ct.target.macOS15,skip_model_load=True);path=art/'norm.mlpackage';cm.save(str(path))
for unit in [ct.ComputeUnit.CPU_ONLY,ct.ComputeUnit.CPU_AND_GPU]:
    model=ct.models.MLModel(str(path),compute_units=unit);got=model.predict({'x':x.numpy(),'divisor':np.array([1280.],np.float32)})
    report['coreml'][unit.name]={n:{'different_elements':int(np.count_nonzero(got[n]!=want.numpy())),'max_abs':float(np.max(np.abs(got[n]-want.numpy())))} for n,want in zip(names,expected)}
(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
