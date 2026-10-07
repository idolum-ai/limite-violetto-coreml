"""Bounded tiny-prefix graph preflight; no Core ML qualification claim."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scripts.audit_torch_candidate import make_candidate,candidate_forward,score_metrics,file_hashes
from portable.model import load_model
torch.set_num_threads(2)
p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);a=p.parse_args()
out=ROOT/'evidence/04d-cpu'/a.attempt;out.mkdir(exist_ok=False)
r={'scope':'pure Torch graph direct prefixes; CPU reference','source_sha256':file_hashes(),'cases':[],'status':'running'}
def save():(out/'results.json').write_text(json.dumps(r,indent=2)+'\n')
save();m=load_model();embed,groups,head=make_candidate(m,'fp16',False,'ordered','ordered','ordered','lookup','source')
ids=json.loads((ROOT/'evidence/01-reference/local-input-validation.json').read_text())['cases'][0]['input_ids']
with torch.inference_mode():
    for length in [1,3,15,16,31,32,33]:
        for group in groups:
            for n,v in group.named_buffers():
                if '_cache_' in n:v.zero_()
        for start in range(0,length,32):_,got,_=candidate_forward(embed,groups,head,ids[start:min(length,start+32)],start,'fp16')
        want=m.scores(m.reference_hidden(torch.tensor([ids[:length]]))[:,-1:])
        record={'length':length,**score_metrics(got,want,151667)};r['cases'].append(record);save();print(length,record['max_abs'],record['passed'],flush=True)
r['status']='complete';r['passed']=all(c['passed'] for c in r['cases']);save()
