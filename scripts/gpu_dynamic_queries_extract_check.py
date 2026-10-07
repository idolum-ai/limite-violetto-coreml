"""Verify extracted dynamic-query functions preserve the tested expressions."""
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from coreml_port.dynamic_queries import prepare,scores

torch.set_num_threads(4)
OUT=ROOT/'evidence/04d-gpu/dynamic-queries-extraction';OUT.mkdir(parents=True,exist_ok=False)
tested=json.loads((ROOT/'evidence/04d-gpu/tiny3-dynamic-group8/summary.json').read_text())
helper=ROOT/'scripts/gpu_dynamic_raw_attention.py'
new=ROOT/'coreml_port/dynamic_queries.py'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert tested['source_sha256']['scripts/gpu_dynamic_raw_attention.py']==sha(helper)
report={'tested_helper_sha256':sha(helper),'new_module_sha256':sha(new),
        'script_sha256':sha(Path(__file__)),'cases':[]}
torch.manual_seed(421)
for start,count in ((0,1),(0,2),(0,3),(0,15),(0,16),(0,32),(1024,2),(33,1)):
    q=torch.randn(1,count,10,128)
    positions=torch.arange(start,start+count,dtype=torch.int32)
    old_rows=torch.minimum(torch.tensor(32,dtype=torch.int32),positions[-1]+1)
    old_q=torch.cat((q,torch.zeros(1,31,10,128)),dim=1)[:,:old_rows]
    old_pos=torch.cat((positions,torch.zeros(31,dtype=torch.int32)))[:old_rows]
    new_q,new_pos=prepare(q,positions)
    keys=torch.randn(1,2048,10,128)
    old_scores=torch.einsum('bthd,bshd->bhts',old_q,keys)*.1
    new_scores=scores(new_q,keys)
    row={'start':start,'count':count,'rows':new_q.shape[1],
         'query_exact':bool(torch.equal(new_q,old_q)),
         'positions_exact':bool(torch.equal(new_pos,old_pos)),
         'scores_exact':bool(torch.equal(new_scores,old_scores))}
    report['cases'].append(row)
    print(row,flush=True)
report['all_exact']=all(c[k] for c in report['cases'] for k in ('query_exact','positions_exact','scores_exact'))
(OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
assert report['all_exact']
