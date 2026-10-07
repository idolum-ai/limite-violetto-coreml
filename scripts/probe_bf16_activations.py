"""Check rounded model activations over an exhaustive bounded BF16 grid."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch,coremltools as ct
from torch import nn
import torch.nn.functional as F
from coreml_port.graphs import Q
from coreml_port.cleanup_guard import install
install();torch.set_num_threads(4)
out=ROOT/'evidence/04d-cpu/activations';art=ROOT/'artifacts/04d-cpu/activations';out.mkdir(parents=True,exist_ok=False);art.mkdir(parents=True,exist_ok=False)
# Every BF16 value in [2^-30, 2^10], both signs, plus zero.
bits=np.arange((127-30)*128,(127+10)*128+1,dtype=np.uint16)
bits=np.concatenate([bits,bits|np.uint16(0x8000),np.array([0],np.uint16)])
x=torch.from_numpy(bits.view(np.int16)).view(torch.bfloat16).float().reshape(1,-1)
class Activations(nn.Module):
    def forward(self,x):return Q(torch.sigmoid(x)),Q(F.silu(x)),Q(F.gelu(x))
names=['sigmoid','silu','gelu']
with torch.inference_mode():expected=Activations()(x);native=[torch.sigmoid(x.bfloat16()).float(),F.silu(x.bfloat16()).float(),F.gelu(x.bfloat16()).float()]
trace=torch.jit.trace(Activations().eval(),x,check_trace=False)
model=ct.convert(trace,inputs=[ct.TensorType(name='x',shape=x.shape,dtype=np.float32)],outputs=[ct.TensorType(name=n,dtype=np.float32) for n in names],compute_precision=ct.precision.FLOAT32,minimum_deployment_target=ct.target.macOS15,skip_model_load=True);path=art/'activations.mlpackage';model.save(str(path));cm=ct.models.MLModel(str(path),compute_units=ct.ComputeUnit.CPU_ONLY);got=cm.predict({'x':x.numpy()})
r={'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'inputs':x.numel(),'range':'all BF16 magnitudes 2^-30 through 2^10, both signs, plus zero','compute':'CPU_ONLY','activations':{}}
for n,want,native_want in zip(names,expected,native):
    actual=got[n];diff=np.flatnonzero(actual.reshape(-1)!=want.numpy().reshape(-1));r['activations'][n]={'graph_vs_native_bf16_differences':int((want!=native_want).sum()),'coreml_vs_graph_differences':len(diff),'max_abs':float(np.abs(actual-want.numpy()).max()),'mismatches':[{'x':float(x.flatten()[i]),'got':float(actual.flatten()[i]),'want':float(want.flatten()[i])} for i in diff[:20]]}
(out/'summary.json').write_text(json.dumps(r,indent=2)+'\n');np.savez_compressed(out/'captures.npz',x=x.numpy(),**got,**{n+'_reference':v.numpy() for n,v in zip(names,expected)});print(json.dumps(r,indent=2),flush=True)
