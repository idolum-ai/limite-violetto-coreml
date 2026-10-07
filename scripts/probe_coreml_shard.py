import sys,json,time,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np,coremltools as ct
from portable.model import load_model
from coreml_port.graphs import Embedding,Group
from scripts.probe_full_model import metrics

p=argparse.ArgumentParser();p.add_argument('--arithmetic',choices=['fp32','ordered'],default='fp32');p.add_argument('--softmax',choices=['native','ordered'],default='native');p.add_argument('--compute',choices=['CPU_ONLY','CPU_AND_GPU','both'],default='both');p.add_argument('--model-dir',default='artifacts/04-coreml/model');p.add_argument('--evidence-dir',default='evidence/04-coreml');a=p.parse_args()
from coreml_port.cleanup_guard import install
install()
if a.arithmetic=='ordered':
    import coreml_port.graphs as graphs
    from coreml_port.ordered_linear import linear
    graphs.linear=linear
if a.softmax=='ordered':
    import coreml_port.graphs as graphs
    from coreml_port.ordered_attention import softmax
    graphs.softmax=softmax
torch.set_num_threads(4);m=load_model();g=Group(m,0).eval();e=Embedding(m)
ids=json.loads((ROOT/'evidence/01-reference/local-input-validation.json').read_text())['cases'][0]['input_ids']
results={}
for units in ([ct.ComputeUnit.CPU_ONLY,ct.ComputeUnit.CPU_AND_GPU] if a.compute=='both' else [getattr(ct.ComputeUnit,a.compute)]):
    print('loading',units.name,flush=True);t=time.time();cm=ct.models.MLModel(str(ROOT/a.model_dir/'0.mlpackage'),compute_units=units);state=cm.make_state();r={'load_seconds':time.time()-t,'comparisons':[]}
    for n,v in g.named_buffers():
        if '_cache_' in n:v.zero_()
    with torch.inference_mode():
        for start,end in [(0,32),(32,33),(33,36)]:
            x,ve=e(torch.tensor([ids[start:end]]));pos=torch.arange(start,end,dtype=torch.int32);args=(x,ve,pos,x.clone(),x.clone(),x.clone());expected=g(*args)[0]
            inputs={n:a.numpy() for n,a in zip(['x','ve','positions','tap0','tap12','tap23'],args)}
            t=time.time();got=cm.predict(inputs,state=state)['y'];r['comparisons'].append({'start':start,'end':end,'seconds':time.time()-t,**metrics(torch.from_numpy(got),expected)})
            print(units.name,r['comparisons'][-1],flush=True)
    results[units.name]=r
    (ROOT/a.evidence_dir/'shard-runtime.json').write_text(json.dumps(results,indent=2)+'\n')
(ROOT/a.evidence_dir/'shard-runtime.json').write_text(json.dumps(results,indent=2)+'\n')
