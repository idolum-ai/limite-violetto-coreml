import json
from pathlib import Path
import torch,numpy as np,coremltools as ct
class M(torch.nn.Module):
 def forward(self,x):
  a=torch.clamp(torch.abs(x),min=1e-30);b=torch.log2(a);e=torch.floor(b)-7;s=torch.pow(2.,e);z=x/s;y=torch.round(z)*s
  return a,b,e,s,z,y
x=torch.tensor([0.,1.,-1.,.1,2.,127.,128.,1024.,1e-8,1e6])
t=torch.jit.trace(M().eval(),x);m=ct.convert(t,inputs=[ct.TensorType(name='x',shape=x.shape,dtype=np.float32)],outputs=[ct.TensorType(name=k,dtype=np.float32) for k in ['a','b','e','s','z','y']],convert_to='mlprogram',compute_precision=ct.precision.FLOAT32,minimum_deployment_target=ct.target.macOS15)
m.save('artifacts/04-coreml/rounding/debug.mlpackage');m=ct.models.MLModel('artifacts/04-coreml/rounding/debug.mlpackage',compute_units=ct.ComputeUnit.CPU_ONLY)
r={k:v.tolist() for k,v in m.predict({'x':x.numpy()}).items()};r['torch']=[v.tolist() for v in M()(x)];print(json.dumps(r));Path('evidence/04-coreml/rounding-debug.json').write_text(json.dumps(r,indent=2))
