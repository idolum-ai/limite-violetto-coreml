"""Replay saved group 8 with dynamic score rows and actual-row value product."""
import hashlib,json,sys,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch,coremltools as ct
import torch.nn.functional as F
from coreml_port.cleanup_guard import install
from coreml_port.ordered_norm import NormWrapped
from portable.model import load_model
from scripts.audit_torch_candidate import make_candidate
from coreml_port.dynamic_queries import prepare,scores as score_fn
import coreml_port.graphs as graphs
install();torch.set_num_threads(4)
BASE=ROOT/'evidence/04d-cpu/long-group-locator';OUT=ROOT/'evidence/04d-gpu/long-layer34/group8-productrows';ART=ROOT/'artifacts/04d-gpu/long-layer34/group8-productrows'
OUT.mkdir(parents=True,exist_ok=False);ART.mkdir(parents=True,exist_ok=False)
payload=dict(np.load(BASE/'first-failure-inputs.npz'));states=dict(np.load(BASE/'first-failure-state-before.npz'))
names=('x','ve','positions','tap0','tap12','tap23','norm_divisor');args=tuple(torch.from_numpy(payload[n]) for n in names)
model=load_model();_,groups,_=make_candidate(model,'fp16',False,'ordered','ordered','ordered','lookup','source')
group=groups[8];del groups,model
def finish(layer,x,residual,q,keys,values,v,positions,slots):
    count=q.shape[1]
    query,pos=prepare(q,positions)
    expanded_keys=keys.repeat_interleave(5,dim=2)
    expanded_values=values.repeat_interleave(5,dim=2).permute(0,2,1,3)
    scores=score_fn(query,expanded_keys)
    mask=slots[None,:]<=pos[:,None]
    if layer.window>=0:mask=mask & (slots[None,:]>=pos[:,None]-layer.window)
    weights=graphs.softmax(scores.masked_fill(~mask[None,None],float('-inf')))
    # Key difference: GEMM sees actual query row count, not the score padding.
    weighted=graphs.Q(torch.matmul(weights[:,:,:count],expanded_values).permute(0,2,1,3))
    vn=graphs.xsa_normalize(v.repeat_interleave(5,dim=2))
    y=graphs.Q(weighted-graphs.Q(layer.alpha*graphs.xsa_dot(weighted,vn)*vn))
    gate=graphs.Q(2*graphs.Q(torch.sigmoid(graphs.Q(graphs.linear(x[...,:128],layer.ag)))))
    y=graphs.Q(y*gate.unsqueeze(-1))
    out=graphs.Q(graphs.linear(y.reshape(1,-1,1280),layer.o))
    mixed=graphs.Q(graphs.Q(layer.resid_attn*residual)+graphs.Q(layer.post_attn*out))
    h=graphs.norm(mixed)
    activated=graphs.Q(graphs.Q(F.silu(graphs.Q(graphs.linear(h,layer.gate))))*graphs.Q(graphs.linear(h,layer.up)))
    mlp=graphs.Q(graphs.linear(activated,layer.down))
    return graphs.Q(graphs.Q(layer.resid_mlp*mixed)+graphs.Q(layer.post_mlp*mlp))
for l in group.layers:l.finish=types.MethodType(finish,l)
wrapped=NormWrapped(group).eval()
def reset():
    for n,b in wrapped.named_buffers():
        if '_cache_' in n:
            suffix=n.replace('.','_').split('_cache_')[-1];kind='k_cache_' if 'k_cache_' in n else 'v_cache_'
            value=next(v for k,v in states.items() if k.endswith(kind+suffix));b.copy_(torch.from_numpy(value))
def compare(a,b):
    d=np.abs(a.astype(np.float64)-b.astype(np.float64));bad=np.argwhere(d!=0)
    return {'different_elements':int(len(bad)),'max_abs':float(d.max()),'first_differences':[{'index':i.tolist(),'actual':float(a[tuple(i)]),'reference':float(b[tuple(i)])} for i in bad[:6]]}
report={'status':'starting','positions':payload['positions'].tolist(),
 'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
 (Path(__file__),ROOT/'coreml_port/dynamic_queries.py',ROOT/'coreml_port/graphs.py',ROOT/'coreml_port/source_exp.py')},
 'input_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
 (BASE/'first-failure-inputs.npz',BASE/'first-failure-state-before.npz')}}
def save():(OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
save();reset()
with torch.inference_mode():torch_y,_=wrapped(*args)
report['torch_vs_reference']=compare(torch_y.numpy(),payload['expected']);save();print('torch',report['torch_vs_reference'],flush=True)
reset();traced=torch.jit.trace(wrapped,args,check_trace=False)
seq=ct.RangeDim(1,32,default=2)
inputs=[ct.TensorType(name='x',shape=(1,seq,1280),dtype=np.float32),ct.TensorType(name='ve',shape=(1,seq,2,128),dtype=np.float32),ct.TensorType(name='positions',shape=(seq,),dtype=np.int32)]
inputs += [ct.TensorType(name=n,shape=(1,seq,1280),dtype=np.float32) for n in ('tap0','tap12','tap23')]
inputs += [ct.TensorType(name='norm_divisor',shape=(1,),dtype=np.float32)]
statespec=[ct.StateType(name=n,wrapped_type=ct.TensorType(shape=b.shape,dtype=np.float16)) for n,b in wrapped.named_buffers() if '_cache_' in n]
converted=ct.convert(traced,inputs=inputs,outputs=[ct.TensorType(name='y',dtype=np.float32),ct.TensorType(name='tap23_out',dtype=np.float32)],states=statespec,compute_precision=ct.precision.FLOAT32,minimum_deployment_target=ct.target.macOS15,skip_model_load=True)
package=ART/'8.mlpackage';converted.save(str(package));report['status']='converted';save()
host=ct.models.MLModel(str(package),compute_units=ct.ComputeUnit.CPU_ONLY);state=host.make_state()
for s in host.get_spec().description.state:
    suffix=s.name.replace('.','_').split('_cache_')[-1];kind='k_cache_' if 'k_cache_' in s.name else 'v_cache_'
    value=next(v for k,v in states.items() if k.endswith(kind+suffix));state.write_state(s.name,value)
got=host.predict({n:payload[n] for n in names},state=state)
report['coreml_vs_reference']=compare(got['y'],payload['expected']);report['coreml_vs_candidate3']=compare(got['y'],payload['actual'])
report['status']='complete';save();np.savez_compressed(OUT/'outputs.npz',y=got['y'],expected=payload['expected'],candidate3=payload['actual'])
print('coreml',report['coreml_vs_reference'],flush=True)
