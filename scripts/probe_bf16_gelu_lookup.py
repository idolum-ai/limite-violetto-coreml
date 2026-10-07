"""Exhaustive bounded lookup validation, including asymmetric tail controls."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np,coremltools as ct
import torch.nn.functional as F
from coreml_port.bf16_gelu import BF16Gelu
from coreml_port.graphs import Q
from coreml_port.cleanup_guard import install
install();torch.set_num_threads(4)
out=ROOT/'evidence/04d-cpu/gelu-lookup';art=ROOT/'artifacts/04d-cpu/gelu-lookup';out.mkdir(parents=True,exist_ok=False);art.mkdir(parents=True,exist_ok=False)
bits=torch.arange((127-30)*128,(127+10)*128+1,dtype=torch.int16);x=bits.view(torch.bfloat16).float();x=torch.cat([x,-x,torch.tensor([0.,2.**-31,-2.**-31,2048.,-2048.])]).reshape(1,-1)
module=BF16Gelu().eval();want=Q(F.gelu(x));expected=module(x);trace=torch.jit.trace(module,x,check_trace=False)
model=ct.convert(trace,inputs=[ct.TensorType(name='x',shape=x.shape,dtype=np.float32)],outputs=[ct.TensorType(name='y',dtype=np.float32)],compute_precision=ct.precision.FLOAT32,minimum_deployment_target=ct.target.macOS15,skip_model_load=True);path=art/'gelu.mlpackage';model.save(str(path));r={'inputs':x.numel(),'torch_differences':int((want!=expected).sum()),'coreml':{}}
for unit in [ct.ComputeUnit.CPU_ONLY,ct.ComputeUnit.CPU_AND_GPU]:
    cm=ct.models.MLModel(str(path),compute_units=unit);got=cm.predict({'x':x.numpy()})['y'];r['coreml'][unit.name]={'different_elements':int(np.count_nonzero(got!=want.numpy())),'max_abs':float(np.abs(got-want.numpy()).max())}
(out/'summary.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2),flush=True)
