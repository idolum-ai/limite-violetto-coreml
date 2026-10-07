"""Test graph-friendly BF16 rounding before using it in real-model graphs."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch import nn
import coremltools as ct

def rounded(x):
    exponent=torch.floor(torch.log2(torch.clamp(torch.abs(x),min=1e-30)))-7
    powers=torch.pow(2.,torch.arange(-128,128,dtype=torch.float32))
    step=powers[(exponent+128).long()]
    return torch.round(x/step)*step

class Round(nn.Module):
    def forward(self,x):return rounded(x)

torch.set_num_threads(4);g=torch.Generator().manual_seed(42)
x=torch.cat([torch.randn(10000,generator=g)*10**e for e in [-8,-3,0,3,6]])
ties=torch.tensor([1.+i/256 for i in range(256)])
x=torch.cat((x,ties,-ties,torch.tensor([0.,1.,-1.,2.,-2.])))
expected=x.bfloat16().float();actual=rounded(x)
out=ROOT/'evidence/04-coreml';art=ROOT/'artifacts/04-coreml/rounding-v2';art.mkdir()
r={'count':len(x),'finite_range':[float(x.min()),float(x.max())],
   'torch_exact_matches':int((actual==expected).sum()),'torch_max_abs':float((actual-expected).abs().max()),
   'scope':'Finite model-scale values and ties; not a general NaN/Inf/subnormal BF16 implementation'}
print(r,flush=True)
trace=torch.jit.trace(Round().eval(),x);model=ct.convert(trace,inputs=[ct.TensorType(name='x',shape=x.shape,dtype=np.float32)],outputs=[ct.TensorType(name='y',dtype=np.float32)],convert_to='mlprogram',compute_precision=ct.precision.FLOAT32,minimum_deployment_target=ct.target.macOS15)
model.save(str(art/'rounding.mlpackage'))
for units in [ct.ComputeUnit.CPU_ONLY,ct.ComputeUnit.CPU_AND_GPU]:
    loaded=ct.models.MLModel(str(art/'rounding.mlpackage'),compute_units=units)
    got=loaded.predict({'x':x.numpy()})['y'];want=expected.numpy()
    r[units.name]={'exact_matches':int((got==want).sum()),'max_abs':float(np.max(np.abs(got-want))),'finite':bool(np.isfinite(got).all())}
(out/'rounding-probe-v2.json').write_text(json.dumps(r,indent=2)+'\n');print(r,flush=True)
