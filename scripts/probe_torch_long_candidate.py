"""Localize long-prefix graph versus converter differences before further fixes."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scripts.audit_torch_candidate import make_candidate,candidate_forward,score_metrics,file_hashes,reference_prefix
from scripts.probe_full_model import metrics
from portable.model import load_model
p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);p.add_argument('--length',type=int,default=1026);a=p.parse_args()
torch.set_num_threads(2);out=ROOT/'evidence/04d-cpu'/a.attempt;out.mkdir(exist_ok=False)
r={'scope':'pure Torch full-capacity FP16-cache source graph vs full-prefix reference','source_sha256':file_hashes(),'status':'starting','length':a.length,'chunks':[]}
def save():(out/'results.json').write_text(json.dumps(r,indent=2)+'\n')
save();m=load_model();embed,groups,head=make_candidate(m,'fp16',False,'ordered','ordered','ordered','lookup','source')
base=json.loads((ROOT/'evidence/01-reference/local-input-validation.json').read_text())['cases'][0]['input_ids'];ids=(base*24)[:a.length];r['input_ids']=ids
with torch.inference_mode():
    for start in range(0,len(ids),32):
        h,got,captured=candidate_forward(embed,groups,head,ids[start:start+32],start,'fp16')
        r['chunks'].append({'start':start,'end':min(start+32,len(ids))});save();print('chunk',start,'done',flush=True)
    want_h,want_captures=reference_prefix(m,ids);want=m.scores(want_h[:,-1:])
    r['scores']=score_metrics(got,want,151667);r['groups']={k:metrics(v[:,-1:],want_captures[k][:,-1:]) for k,v in captured.items() if k in want_captures};r['hidden']=metrics(h[:,-1:],want_h[:,-1:])
r['status']='complete';r['passed']=r['scores']['passed'];save();print('scores',r['scores'],flush=True)
