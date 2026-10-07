"""Uninstrumented ordered RMS check over independent inputs and model widths."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch,coremltools as ct
from torch import nn
from coreml_port import graphs
from coreml_port.ordered_norm import raw_norm
from coreml_port.cleanup_guard import install
install();torch.set_num_threads(4)
out=ROOT/'evidence/04d-cpu/ordered-norm-shapes';art=ROOT/'artifacts/04d-cpu/ordered-norm-shapes';out.mkdir(parents=True,exist_ok=False);art.mkdir(parents=True,exist_ok=False)
class Norm(nn.Module):
    def forward(self,x,divisor):return graphs.Q(raw_norm(x,divisor))
result=[];generator=torch.Generator().manual_seed(20260922)
for width,heads in [(1280,None),(128,10)]:
    shape=(1,3,width) if heads is None else (1,3,heads,width)
    sample=torch.randn(shape,generator=generator);divisor=torch.tensor([float(width)])
    traced=torch.jit.trace(Norm().eval(),(sample,divisor),check_trace=False)
    flexible=(1,ct.RangeDim(1,32,default=3),*shape[2:]);name=str(width)
    cm=ct.convert(traced,inputs=[ct.TensorType(name='x',shape=flexible,dtype=np.float32),ct.TensorType(name='divisor',shape=(1,),dtype=np.float32)],outputs=[ct.TensorType(name='y',dtype=np.float32)],compute_precision=ct.precision.FLOAT32,minimum_deployment_target=ct.target.macOS15,skip_model_load=True);path=art/(name+'.mlpackage');cm.save(str(path))
    for unit in [ct.ComputeUnit.CPU_ONLY,ct.ComputeUnit.CPU_AND_GPU]:
        model=ct.models.MLModel(str(path),compute_units=unit)
        for count in [1,3,32]:
            for scale in [2.**-12,1.,4096.]:
                shape=(1,count,*shape[2:]);x=torch.randn(shape,generator=generator).bfloat16().float()*scale
                with torch.inference_mode():want=graphs.norm(x).numpy()
                got=model.predict({'x':x.numpy(),'divisor':divisor.numpy()})['y'];different=int(np.count_nonzero(got!=want))
                r={'width':width,'shape':list(shape),'scale':scale,'compute':unit.name,'different_elements':different,'max_abs':float(np.max(np.abs(got-want)))};result.append(r);print(r,flush=True)
                (out/'summary.json').write_text(json.dumps({'comparisons':result,'all_exact':all(v['different_elements']==0 for v in result)},indent=2)+'\n')
