"""Isolate a single long forward call, without loading the Torch reference."""
import faulthandler,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));faulthandler.enable()
import torch,numpy as np
from coreml_port.runtime import Pipeline
cm=Pipeline();ids=torch.randint(100,10000,(1026,),generator=torch.Generator().manual_seed(20260922)).tolist();req=cm.request();r={'compute':'CPU_ONLY','completed':[]};t=time.time()
for chunk in [ids[:1024],ids[1024:1025],ids[1025:], [100]*1022]:
    s=cm.forward(chunk,req);r['completed'].append({'length':req.length,'finite':bool(np.isfinite(s).all())});print(r['completed'][-1],flush=True)
req.close();r['elapsed_seconds']=time.time()-t;(ROOT/'evidence/04-coreml/longprefill-isolation.json').write_text(json.dumps(r,indent=2)+'\n')
