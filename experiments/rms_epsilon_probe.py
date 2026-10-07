"""Reproduce the input-dtype versus opmath RMS epsilon discrepancy on CPU."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F

results=[]
for dtype in (torch.bfloat16,torch.float16,torch.float32):
    x=torch.tensor([[0.001,0.002,0.003,0.004]],dtype=dtype)
    default=F.rms_norm(x,(4,))
    declared=F.rms_norm(x,(4,),eps=torch.finfo(dtype).eps)
    opmath=F.rms_norm(x,(4,),eps=torch.finfo(torch.float32).eps)
    results.append({'dtype':str(dtype),'default':default.float().tolist(),
        'explicit_dtype_epsilon':declared.float().tolist(),'explicit_fp32_epsilon':opmath.float().tolist(),
        'default_vs_dtype_eps_max':float((default-declared).abs().max()),
        'default_vs_fp32_eps_max':float((default-opmath).abs().max())})
target=Path(__file__).resolve().parents[1]/'evidence/02-components/rms-epsilon-probe.json'
target.write_text(json.dumps({'torch':torch.__version__,'device':'CPU','results':results},indent=2)+'\n')
print(target.read_text())
