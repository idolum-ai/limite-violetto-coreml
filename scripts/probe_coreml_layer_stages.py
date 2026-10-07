"""Expose first-layer rounding sites to localize residual Core ML differences.

Diagnostic outputs can affect compiler fusion; this does not replace an
unmodified end-to-end gate run.
"""
import json,sys,inspect,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np,coremltools as ct
from torch import nn
import coreml_port.graphs as graphs
from coreml_port.ordered_linear import linear
from coreml_port.cleanup_guard import install
from portable.model import load_model
from scripts.probe_full_model import metrics
p=argparse.ArgumentParser();p.add_argument('--softmax',choices=['native','ordered'],default='native');p.add_argument('--evidence-dir',default='evidence/04c-precision');p.add_argument('--artifact-dir',default='artifacts/04c-precision');p.add_argument('--compute',choices=['CPU_ONLY','CPU_AND_GPU','both'],default='both');a=p.parse_args()
evidence=ROOT/a.evidence_dir;artifact=ROOT/a.artifact_dir;evidence.mkdir(parents=True,exist_ok=True);artifact.mkdir(parents=True,exist_ok=True)
if a.softmax=='ordered':
    from coreml_port.ordered_attention import softmax
    graphs.softmax=softmax
install();torch.set_num_threads(4);graphs.linear=linear;m=load_model();ids=torch.tensor([json.loads((ROOT/'evidence/01-reference/local-input-validation.json').read_text())['cases'][0]['input_ids'][:32]])
with torch.inference_mode():x,ve=graphs.Embedding(m)(ids)
group=graphs.Group(m,0).eval();original_q=graphs.Q
class Diagnostic(nn.Module):
    def __init__(self):super().__init__();self.group=group
    def forward(self,x,ve,positions,tap0,tap12,tap23):
        records=[];sites=[]
        def capture(v):
            result=original_q(v);records.append(result)
            frame=inspect.currentframe().f_back;caller=frame;layer=None
            while frame:
                obj=frame.f_locals.get('self')
                if hasattr(obj,'i'):layer=obj.i;break
                frame=frame.f_back
            sites.append({'function':caller.f_code.co_name,'line':caller.f_lineno,'layer':layer})
            return result
        graphs.Q=capture
        try:y,tap=self.group(x,ve,positions,tap0,tap12,tap23)
        finally:graphs.Q=original_q
        self.sites=sites
        return (y,*records)
g=Diagnostic().eval();args=(x,ve,torch.arange(32,dtype=torch.int32),x.clone(),x.clone(),x.clone())
with torch.inference_mode():expected=g(*args)
sites=list(g.sites)
(evidence/'rounding-site-labels.json').write_text(json.dumps(sites,indent=2)+'\n')
for n,v in g.named_buffers():
    if '_cache_' in n:v.zero_()
trace=torch.jit.trace(g,args,check_trace=False);names=['y']+[f'round_{i}' for i in range(len(expected)-1)]
inputs=[ct.TensorType(name=n,shape=v.shape,dtype=np.int32 if n=='positions' else np.float32) for n,v in zip(['x','ve','positions','tap0','tap12','tap23'],args)]
states=[ct.StateType(name=n,wrapped_type=ct.TensorType(shape=v.shape,dtype=np.float16)) for n,v in g.named_buffers() if '_cache_' in n]
path=artifact/'rounding-sites.mlpackage'
if not path.exists():
    model=ct.convert(trace,inputs=inputs,outputs=[ct.TensorType(name=n,dtype=np.float32) for n in names],states=states,compute_precision=ct.precision.FLOAT32,minimum_deployment_target=ct.target.macOS15,skip_model_load=True);model.save(str(path))
r={}
for units in ([ct.ComputeUnit.CPU_ONLY,ct.ComputeUnit.CPU_AND_GPU] if a.compute=='both' else [getattr(ct.ComputeUnit,a.compute)]):
    cm=ct.models.MLModel(str(path),compute_units=units);got=cm.predict({n:v.numpy() for n,v in zip(['x','ve','positions','tap0','tap12','tap23'],args)},state=cm.make_state());result=[]
    keys=[o.name for o in cm.get_spec().description.output]
    for n,key,want in zip(names,keys,expected):
        v=metrics(torch.from_numpy(got[key]),want);v.update(name=n,shape=list(want.shape),different_elements=int(np.count_nonzero(got[key]!=want.numpy())));result.append(v)
    r[units.name]=result;print(units.name,'first differences',[(x['name'],x['max_abs'],x['different_elements']) for x in result if x['different_elements']][:12],flush=True)
    (evidence/'rounding-sites.json').write_text(json.dumps(r,indent=2)+'\n')
