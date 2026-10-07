"""Locate first converted group difference on a native full-prefix fixture."""
import argparse,gc,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np,coremltools as ct
from portable.model import load_model,rms
from scripts.audit_torch_candidate import reference_prefix
from scripts.probe_full_model import metrics
from coreml_port.cleanup_guard import install
from coreml_port.package_identity import verify_candidate_identity
p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);p.add_argument('--model-dir',required=True);p.add_argument('--identity-manifest',required=True);a=p.parse_args()
install();torch.set_num_threads(4);out=ROOT/'evidence/04d-cpu'/a.attempt;out.mkdir(exist_ok=False)
r={'status':'starting','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'package_identity':verify_candidate_identity(ROOT/a.model_dir,ROOT/a.identity_manifest),'groups':[]}
def save():(out/'results.json').write_text(json.dumps(r,indent=2)+'\n')
save();m=load_model();base=json.loads((ROOT/'evidence/01-reference/local-input-validation.json').read_text())['cases'][0]['input_ids'];ids=(base*24)[:1026];r['ids']=ids;tids=torch.tensor([ids]);extra={}
def keep22(module,args,value):extra['layer_22']=value.clone()
handle=m.backbone.layers[22].register_forward_hook(keep22)
with torch.inference_mode():
    hidden,captures=reference_prefix(m,ids);captures.update(extra);captures['embedding_norm']=rms(m.backbone.embed_tokens(tids));captures['ve']=m.backbone._value_embeddings(tids);captures['scores']=m.scores(hidden[:,-1:])
handle.remove();arrays={k:v.float().numpy().copy() for k,v in captures.items()};np.savez_compressed(out/'reference.npz',**arrays)
del m,captures,extra,hidden;gc.collect();r['status']='reference captured; comparing groups';save();print('reference captured',flush=True)
for g in range(12):
    cm=ct.models.MLModel(str(ROOT/a.model_dir/f'{g}.mlpackage'),compute_units=ct.ComputeUnit.CPU_ONLY);state=cm.make_state();names=[s.name for s in cm.get_spec().description.state]
    x=arrays['embedding_norm' if g==0 else f'layer_{4*g-1}'];target=arrays['final_hidden' if g==11 else f'layer_{4*g+3}'];row={'group':g,'steps':[]};r['groups'].append(row)
    for start in range(0,len(ids),32):
        end=min(start+32,len(ids));inputs={'x':x[:,start:end].copy(),'ve':arrays['ve'][:,start:end].copy(),'positions':np.arange(start,end,dtype=np.int32),'tap0':arrays['embedding_norm'][:,start:end].copy(),'tap12':arrays['layer_11'][:,start:end].copy(),'tap23':arrays['layer_22'][:,start:end].copy(),'norm_divisor':np.array([1280.],np.float32)}
        before={n:state.read_state(n).copy() for n in names};got=cm.predict(inputs,state=state)['y'];expected=target[:,start:end];metric=metrics(torch.from_numpy(got),torch.from_numpy(expected));different=int(np.count_nonzero(got!=expected));row['steps'].append({'start':start,'end':end,'different_elements':different,**metric});save();print('group',g,'end',end,'different',different,flush=True)
        if different:
            np.savez_compressed(out/'first-failure-inputs.npz',**inputs,actual=got,expected=expected)
            np.savez_compressed(out/'first-failure-state-before.npz',**before)
            r['status']='first group/step difference captured';r['first_failure']={'group':g,'start':start,'end':end};save();raise SystemExit(0)
    del cm,state;gc.collect()
r['status']='all groups exact with native reference inputs';save()
