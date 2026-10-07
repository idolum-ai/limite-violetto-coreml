"""Canonical shard-0 CPU replay of actual integrated dynamic-product converter output."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch,coremltools as ct
from coreml_port.cleanup_guard import install
from coreml_port.ordered_norm import NormWrapped
from scripts.audit_torch_candidate import make_candidate
from portable.model import load_model
install();torch.set_num_threads(4)
OUT=ROOT/'evidence/04d-gpu/long-layer34/integrated-shard0'
PACKAGE=ROOT/'artifacts/04d-gpu/long-layer34/integrated-shard0/0.mlpackage'
cases=json.loads((ROOT/'evidence/01-reference/local-input-validation.json').read_text())['cases']
model=load_model();embedding,groups,_=make_candidate(model,'fp16',False,'ordered','ordered','ordered','lookup','native')
emb=NormWrapped(embedding).eval();group=NormWrapped(groups[0]).eval();del model,groups
divisor=torch.tensor([1280.]);host=ct.models.MLModel(str(PACKAGE),compute_units=ct.ComputeUnit.CPU_ONLY)
report={'status':'starting','compute_units':'CPU_ONLY','package':str(PACKAGE.relative_to(ROOT)),
 'package_files_sha256':{str(p.relative_to(PACKAGE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in PACKAGE.rglob('*') if p.is_file()},
 'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'prompts':[]}
def save():(OUT/'gate.json').write_text(json.dumps(report,indent=2)+'\n')
def compare(a,b):
    d=np.abs(a.astype(np.float64)-b.astype(np.float64));bad=np.argwhere(d!=0)
    return {'different_elements':int(len(bad)),'max_abs':float(d.max()),
      'first_differences':[{'index':i.tolist(),'actual':float(a[tuple(i)]),'reference':float(b[tuple(i)])} for i in bad[:6]]}
save()
for index,case in enumerate(cases):
    ids=case['input_ids'];steps=[(start,ids[start:start+32],'prefill') for start in range(0,len(ids),32)]
    steps.append((len(ids),[16],'decode1'))
    state=host.make_state()
    for n,b in group.named_buffers():
        if '_cache_' in n:b.zero_()
    prompt={'index':index,'prompt_length':len(ids),'steps':[]};report['prompts'].append(prompt);save()
    for start,chunk,kind in steps:
        with torch.inference_mode():
            x,ve=emb(torch.tensor([chunk],dtype=torch.int32),divisor)
            positions=torch.arange(start,start+len(chunk),dtype=torch.int32)
            args=(x,ve,positions,x.clone(),x.clone(),x.clone(),divisor)
            reference_y,reference_tap=group(*args)
        payload={n:v.numpy() for n,v in zip(('x','ve','positions','tap0','tap12','tap23','norm_divisor'),args)}
        got=host.predict(payload,state=state)
        row={'kind':kind,'start':start,'length':len(chunk),'y':compare(got['y'],reference_y.numpy()),
             'tap23_out':compare(got['tap23_out'],reference_tap.numpy())}
        prompt['steps'].append(row);save();print(index,kind,start,row['y']['different_elements'],flush=True)
report['all_exact']=all(s[k]['different_elements']==0 for p in report['prompts'] for s in p['steps'] for k in ('y','tap23_out'))
report['status']='complete';save();print('all_exact',report['all_exact'],flush=True)
