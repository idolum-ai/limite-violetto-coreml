"""Bounded layer-34 long-prefix attention operator replay."""
import hashlib, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np, torch, coremltools as ct
from torch import nn
from coreml_port.cleanup_guard import install
from coreml_port.ordered_norm import NormWrapped
from coreml_port.ordered_attention import softmax
from scripts.audit_torch_candidate import make_candidate
from portable.model import load_model
import coreml_port.graphs as graphs

install();torch.set_num_threads(4)
BASE=ROOT/'evidence/04d-cpu/long-group-locator'
OUT=ROOT/'evidence/04d-gpu/long-layer34';ART=ROOT/'artifacts/04d-gpu/long-layer34'
OUT.mkdir(parents=True,exist_ok=False);ART.mkdir(parents=True,exist_ok=False)
payload=dict(np.load(BASE/'first-failure-inputs.npz'));states=dict(np.load(BASE/'first-failure-state-before.npz'))
names=('x','ve','positions','tap0','tap12','tap23','norm_divisor')
args=tuple(torch.from_numpy(payload[n]) for n in names)
model=load_model();_,groups,_=make_candidate(model,'fp16',False,'ordered','ordered','ordered','lookup','source')
group=NormWrapped(groups[8]).eval();del groups,model
for name,buf in group.named_buffers():
    if '_cache_' in name:
        suffix=name.replace('.','_').split('_cache_')[-1]
        kind='k_cache_' if 'k_cache_' in name else 'v_cache_'
        value=next(v for k,v in states.items() if k.endswith(kind+suffix))
        buf.copy_(torch.from_numpy(value))
saved={};original=graphs.Layer.finish
def capture(self,x,residual,q,keys,values,v,positions,slots):
    if self.i==34:
        with torch.inference_mode():
            qp=torch.cat((q,torch.zeros(1,31,10,128)),dim=1)[:,:32].permute(0,2,1,3)
            k=keys.repeat_interleave(5,dim=2).permute(0,2,3,1)
            val=values.repeat_interleave(5,dim=2).permute(0,2,1,3)
            pos=torch.cat((positions,torch.zeros(31,dtype=torch.int32)))[:32]
            mask=slots[None,:]<=pos[:,None]
            if self.window>=0:mask=mask & (slots[None,:]>=pos[:,None]-self.window)
            scores=(qp@k)*.1
            logits=scores.masked_fill(~mask[None,None],float('-inf'))
            weights=graphs.softmax(logits)
            raw=weights@val
            rounded=graphs.Q(raw)
            saved.update(q=qp.detach().numpy(),k=k.detach().numpy(),val=val.detach().numpy(),
                scores=scores.detach().numpy(),logits=logits.detach().numpy(),
                weights=weights.detach().numpy(),raw=raw.detach().numpy(),rounded=rounded.detach().numpy(),
                mask=mask.numpy(),positions=positions.numpy())
    return original(self,x,residual,q,keys,values,v,positions,slots)
graphs.Layer.finish=capture
try:
    with torch.inference_mode():y,_=group(*args)
finally:graphs.Layer.finish=original
assert np.array_equal(y.numpy(),payload['expected']),'Torch replay must match saved native result'
np.savez_compressed(OUT/'reference.npz',**saved)

def metric(actual,want):
    delta=np.abs(actual.astype(np.float64)-want.astype(np.float64));bad=np.argwhere(delta!=0)
    return {'different_elements':int(len(bad)),'max_abs':float(delta.max()),
            'target_actual':float(actual[0,9,1,26]),'target_reference':float(want[0,9,1,26]),
            'first_differences':[{'index':i.tolist(),'actual':float(actual[tuple(i)]),'reference':float(want[tuple(i)])} for i in bad[:8]]}
class Product(nn.Module):
    def forward(self,weights,val):
        raw=weights@val
        return raw,graphs.Q(raw)
def run(module,input_names,output_names,arrays,stem):
    vals=tuple(torch.from_numpy(saved[n].copy()) for n in input_names)
    trace=torch.jit.trace(module.eval(),vals,check_trace=False)
    converted=ct.convert(trace,inputs=[ct.TensorType(name=n,shape=v.shape,dtype=np.float32) for n,v in zip(input_names,vals)],
        outputs=[ct.TensorType(name=n,dtype=np.float32) for n in output_names],
        compute_precision=ct.precision.FLOAT32,minimum_deployment_target=ct.target.macOS15,skip_model_load=True)
    package=ART/f'{stem}.mlpackage';converted.save(str(package))
    cm=ct.models.MLModel(str(package),compute_units=ct.ComputeUnit.CPU_ONLY)
    got=cm.predict({n:saved[n] for n in input_names})
    arrays.update({f'{stem}_{n}':got[n] for n in output_names})
    return {n:got[n] for n in output_names}

report={'status':'starting','positions':saved['positions'].tolist(),'shape':{n:list(saved[n].shape) for n in ('q','k','val','scores','weights')},
    'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [ROOT/'coreml_port/graphs.py',ROOT/'coreml_port/source_exp.py',ROOT/'coreml_port/ordered_attention.py',ROOT/'coreml_port/ordered_norm.py',ROOT/'coreml_port/ordered_linear.py']},
    'input_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [BASE/'first-failure-inputs.npz',BASE/'first-failure-state-before.npz']}}
def save(): (OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
save();arrays={}
product=run(Product(),('weights','val'),('raw','rounded'),arrays,'product')
report['product']={n:metric(product[n],saved[n]) for n in ('raw','rounded')};save()
print('PRODUCT',report['product'],flush=True)

class Softmax(nn.Module):
    def forward(self,logits):return graphs.softmax(logits)
got=run(Softmax(),('logits',),('weights',),arrays,'softmax')
report['softmax']={'different_elements':int(np.count_nonzero(got['weights']!=saved['weights'])),
    'max_abs':float(np.max(np.abs(got['weights'].astype(np.float64)-saved['weights'].astype(np.float64))))}
with torch.inference_mode():
    mixed_raw=torch.from_numpy(got['weights'])@torch.from_numpy(saved['val'])
    mixed_q=graphs.Q(mixed_raw).numpy()
report['softmax']['mixed_rounded']=metric(mixed_q,saved['rounded']);arrays['mixed_rounded']=mixed_q
save();print('SOFTMAX',report['softmax'],flush=True)

class Scores(nn.Module):
    def forward(self,q,k):return (q@k)*.1
got=run(Scores(),('q','k'),('scores',),arrays,'scores')
report['scores']={'different_elements':int(np.count_nonzero(got['scores']!=saved['scores'])),
    'max_abs':float(np.max(np.abs(got['scores'].astype(np.float64)-saved['scores'].astype(np.float64))))}
with torch.inference_mode():
    logits=torch.from_numpy(got['scores']).masked_fill(~torch.from_numpy(saved['mask'])[None,None],float('-inf'))
    mixed_weights=graphs.softmax(logits)
    mixed_q=graphs.Q(mixed_weights@torch.from_numpy(saved['val'])).numpy()
report['scores']['mixed_rounded']=metric(mixed_q,saved['rounded']);arrays['scores_mixed_rounded']=mixed_q
report['status']='complete';save();np.savez_compressed(OUT/'outputs.npz',**arrays)
print('SCORES',report['scores'],flush=True)
