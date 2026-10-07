"""Whole-model delayed cleanup and original long-context pattern, guarded."""
import faulthandler,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));faulthandler.enable()
import torch,numpy as np
from portable.model import load_model
from coreml_port.runtime import Pipeline
from coreml_port.cleanup_guard import stats
from scripts.probe_full_model import metrics
torch.set_num_threads(4);cm=Pipeline();m=load_model();ids=torch.randint(100,10000,(1026,),generator=torch.Generator().manual_seed(20260922)).tolist();records=[]
for trial in range(3):
    req=cm.request();begin=time.time()
    cm.forward(ids[:1024],req);scores=[]
    with torch.inference_mode():
        for end in [1025,1026]:
            got=cm.forward([ids[end-1]],req);want=m.scores(m.reference_hidden(torch.tensor([ids[:end]]))[:,-1:]);scores.append(metrics(torch.from_numpy(got),want))
    cm.forward([100]*1022,req)
    try:cm.forward([100],req);overflow=False
    except ValueError:overflow=req.length==2048
    req.close();time.sleep(.5);r={'trial':trial,'seconds':time.time()-begin,'scores':scores,'overflow_rejected':overflow,'closed':req.closed,'guard':stats()};records.append(r);print(r,flush=True)
    (ROOT/'evidence/04b-reliability/long-context-stress.json').write_text(json.dumps(records,indent=2)+'\n')
print('completed',flush=True)
