"""Bounded empirical four-lane Torch prefix audit; not a Core ML gate."""
from pathlib import Path
import hashlib,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
import coreml_port.graphs as graphs
from coreml_port import dynamic_queries,paired_product
from scripts.gpu_four_lane_scores import scores as four_lane_scores
from scripts.audit_torch_candidate import make_candidate,candidate_forward,score_metrics
from scripts.probe_full_model import metrics
from portable.model import load_model
out=ROOT/'evidence/04d-gpu/four-lane-prefix-preflight';out.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(2)
source=ROOT/'evidence/03b-cache/cpu-partitions-v3.json'
by_name={v['name']:v for v in json.loads(source.read_text())['cases']}
fixtures={k:by_name[k]['input_ids'][0][:16] for k in ['tiny_33','tiny_15','irregular_127']}
fixtures['fresh_seed_20260923']=torch.randint(100,151667,(16,),generator=torch.Generator().manual_seed(20260923)).tolist()
r={'status':'running','scope':'pure Torch composed four-lane adjacent empirical policy, not Core ML qualification','fixtures':fixtures,'comparisons':[],'thresholds_changed':False,'sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['scripts/gpu_probe_four_lane_prefixes.py','scripts/gpu_four_lane_scores.py','coreml_port/paired_product.py','coreml_port/dynamic_queries.py','coreml_port/graphs.py','evidence/03-portable/ACCEPTANCE.json']}}
def save():(out/'results.json').write_text(json.dumps(r,indent=2)+'\n')
save();start=time.time();model=load_model();embed,groups,head=make_candidate(model,'fp16',False,'ordered','ordered','ordered','lookup','source')
graphs.prepare_queries=dynamic_queries.prepare;graphs.attention_product=paired_product.product;graphs.attention_scores=lambda q,k:four_lane_scores(q,k,'adjacent')
with torch.inference_mode():
 for name,ids in fixtures.items():
  for length in range(1,len(ids)+1):
   for g in groups:
    for k,v in g.named_buffers():
     if '_cache_' in k:v.zero_()
   native=model.reference_hidden(torch.tensor([ids[:length]]))
   h,s,_=candidate_forward(embed,groups,head,ids[:length],0,'fp16')
   sm=score_metrics(s,model.scores(native[:,-1:]),151667);hm=metrics(h,native)
   passed=sm['passed'] and hm['relative_rmse']<=.01 and hm['cosine']>=.995
   rec={'fixture':name,'length':length,'scores':sm,'hidden':hm,'pass':passed};r['comparisons'].append(rec);save();print(name,length,passed,sm['max_abs'],flush=True)
r.update(status='complete',all_pass=all(x['pass'] for x in r['comparisons']),seconds=time.time()-start);save()
