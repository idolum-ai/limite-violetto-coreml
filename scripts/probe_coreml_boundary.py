"""Complete Core ML local-window crossing and maximum-position execution."""
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from coreml_port.runtime import Pipeline
from portable.model import load_model
from scripts.probe_full_model import metrics
torch.set_num_threads(4);cm=Pipeline();m=load_model();req=cm.request();g=torch.Generator().manual_seed(20260922);ids=torch.randint(100,10000,(1026,),generator=g).tolist();r={'compute':'CPU_ONLY','comparisons':[]};t=time.time();print("loaded",flush=True)
with torch.inference_mode():
    for offset in range(0,1024,32):
        cm.forward(ids[offset:offset+32],req);print("prefill",req.length,flush=True)
    for end in [1025,1026]:
        got=cm.forward([ids[end-1]],req);want=m.scores(m.reference_hidden(torch.tensor([ids[:end]]))[:,-1:]);v=metrics(torch.from_numpy(got),want);v.update(position=end-1,pass_score=v['max_abs']<=.04);r['comparisons'].append(v);print(v,flush=True)
    # Reach the true capacity through contiguous writes; no fabricated length.
    while req.length<2048:
        cm.forward([100]*min(32,2048-req.length),req);print('capacity',req.length,flush=True)
    r['capacity_finite']=req.length==2048
    try:cm.forward([100],req);r['overflow_rejected']=False
    except ValueError:r['overflow_rejected']=req.length==2048 and req.valid
req.close();r['elapsed_seconds']=time.time()-t;(ROOT/'evidence/04-coreml/boundary.json').write_text(json.dumps(r,indent=2)+'\n');print(r,flush=True)
