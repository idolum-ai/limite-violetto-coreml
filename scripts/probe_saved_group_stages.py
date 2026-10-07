"""Replay a captured group failure and expose BF16 rounding sites."""
import argparse,gc,hashlib,inspect,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch,coremltools as ct
from torch import nn
from coreml_port.cleanup_guard import install
from scripts.audit_torch_candidate import make_candidate
from coreml_port.ordered_norm import NormWrapped
from portable.model import load_model
import coreml_port.graphs as graphs
p=argparse.ArgumentParser();p.add_argument('--capture',required=True);p.add_argument('--attempt',required=True);a=p.parse_args()
install();torch.set_num_threads(4);source=ROOT/a.capture;out=ROOT/'evidence/04d-cpu'/a.attempt;art=ROOT/'artifacts/04d-cpu'/a.attempt;out.mkdir(exist_ok=False);art.mkdir(exist_ok=False)
meta=json.loads((source/'results.json').read_text());index=meta['first_failure']['group'];saved=dict(np.load(source/'first-failure-inputs.npz'));states=dict(np.load(source/'first-failure-state-before.npz'))
names=['x','ve','positions','tap0','tap12','tap23','norm_divisor'];inputs={n:saved[n] for n in names}
def match_state(n):
    suffix=n.replace('.','_').split('_cache_')[-1]
    kind='k_cache_' if 'k_cache_' in n else 'v_cache_'
    return next(v for k,v in states.items() if k.endswith(kind+suffix))
def write_states(cm,state):
    for s in cm.get_spec().description.state:state.write_state(s.name,match_state(s.name))
def compare(x,y):
    bad=np.argwhere(x!=y);return {'different_elements':int(len(bad)),'max_abs':float(np.max(np.abs(x-y))),'first_differences':[{'index':i.tolist(),'actual':float(x[tuple(i)]),'expected':float(y[tuple(i)])} for i in bad[:5]]}
r={'capture':str(source),'group':index,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'status':'starting'}
def save():(out/'results.json').write_text(json.dumps(r,indent=2)+'\n')
save();old=ct.models.MLModel(str(ROOT/'artifacts/04d-cpu/full-candidate-3'/f'{index}.mlpackage'),compute_units=ct.ComputeUnit.CPU_ONLY);state=old.make_state();write_states(old,state);replay=old.predict(inputs,state=state)['y'];r['ordinary_replay_vs_saved_actual']=compare(replay,saved['actual']);save();del old,state;gc.collect()
if r['ordinary_replay_vs_saved_actual']['different_elements']:raise RuntimeError('Ordinary replay differs; do not interpret stage probe.')
m=load_model();_,groups,_=make_candidate(m,'fp16',False,'ordered','ordered','ordered','lookup','source');group=groups[index];del m,groups;gc.collect();base_q=graphs.Q
class Diagnostic(nn.Module):
    def __init__(self):super().__init__();self.inner=NormWrapped(group)
    def forward(self,*args):
        records=[];sites=[]
        def capture(value):
            result=base_q(value);records.append(result+0.)
            caller=inspect.currentframe().f_back;frame=caller;layer=None
            while frame:
                obj=frame.f_locals.get('self')
                if hasattr(obj,'i'):layer=obj.i;break
                frame=frame.f_back
            sites.append({'function':caller.f_code.co_name,'line':caller.f_lineno,'layer':layer});return result
        graphs.Q=capture
        try:y,tap=self.inner(*args)
        finally:graphs.Q=base_q
        self.sites=sites;return (y,*records)
graph=Diagnostic().eval();args=tuple(torch.from_numpy(inputs[n]) for n in names)
def reset():
    for n,v in graph.named_buffers():
        if '_cache_' in n:v.copy_(torch.from_numpy(match_state(n)))
with torch.inference_mode():reset();expected=graph(*args)
labels=graph.sites;r['torch_with_actual_state_vs_native']=compare(expected[0].numpy(),saved['expected']);r['torch_with_actual_state_vs_coreml']=compare(expected[0].numpy(),replay);save()
reset();traced=torch.jit.trace(graph,args,check_trace=False);output_names=['y']+[f'round_{i}' for i in range(len(expected)-1)]
model=ct.convert(traced,inputs=[ct.TensorType(name=n,shape=inputs[n].shape,dtype=np.int32 if n=='positions' else np.float32) for n in names],outputs=[ct.TensorType(name=n,dtype=np.float32) for n in output_names],states=[ct.StateType(name=n,wrapped_type=ct.TensorType(shape=v.shape,dtype=np.float16)) for n,v in graph.named_buffers() if '_cache_' in n],compute_precision=ct.precision.FLOAT32,minimum_deployment_target=ct.target.macOS15,skip_model_load=True)
package=art/'stages.mlpackage';model.save(str(package));cm=ct.models.MLModel(str(package),compute_units=ct.ComputeUnit.CPU_ONLY);state=cm.make_state();write_states(cm,state);got=cm.predict(inputs,state=state);keys=[s.name for s in cm.get_spec().description.output]
rows=[];arrays={}
for j,(key,want) in enumerate(zip(keys,expected)):
    row={'output':key,**compare(got[key],want.numpy())}
    if j:row['site']=labels[j-1]
    rows.append(row);arrays[key+'_actual']=got[key];arrays[key+'_expected']=want.numpy()
r['stages']=rows;r['status']='complete';save();np.savez_compressed(out/'stages.npz',**arrays);print('first differences',[(v['output'],v.get('site'),v['different_elements']) for v in rows if v['different_elements']][:12],flush=True)
