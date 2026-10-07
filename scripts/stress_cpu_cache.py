"""Independent partitions and tiny prefixes for the sixteen-row CPU candidate."""
import hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from portable.model import load_model
from probe_full_model import metrics

torch.set_num_threads(4);torch.manual_seed(913)
model=load_model('cpu')
out=ROOT/'evidence/03b-cache/cpu-partitions-v3.json'
result={'seed':913,'attention_min_query_rows':model.attention_min_query_rows,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cases':[]}
def save():out.write_text(json.dumps(result,indent=2)+'\n')
sequence=torch.randint(0,151643,(1,127))
cases=[(f'tiny_{n}',sequence[:,:n],[1]*n) for n in [1,2,3,4,7,8,9,15,17,31,33]]
cases.append(('irregular_127',sequence,[1,2,4,7,16,32,65]))
for name,ids,parts in cases:
    t=time.perf_counter();reference=model.reference_hidden(ids);plain=model(ids)
    cache=model.new_cache();start=0;outputs=[];matched=[]
    for length in parts:
        current=model(ids[:,start:start+length],cache);outputs.append(current);start+=length
        current_reference=model.reference_hidden(ids[:,:start])[:,-length:]
        hm=metrics(current_reference,current);sm=metrics(model.scores(current_reference[:,-1:]),model.scores(current[:,-1:]))
        matched.append({'prefix_length':start,'hidden':hm,'scores':sm,'passed':hm['relative_rmse']<=.01 and hm['cosine']>=.995 and sm['max_abs']<=.04})
    cached=torch.cat(outputs,dim=1)
    h=metrics(reference,cached);s=metrics(model.scores(reference[:,-1:]),model.scores(cached[:,-1:]))
    entry={'name':name,'input_ids':ids.tolist(),'partitions':parts,'full_prefix_exact':torch.equal(reference,plain),'cached_hidden':h,'cached_scores':s,'cached_length':cache.length,'seconds':time.perf_counter()-t,
           'matched_prefix_comparisons':matched,'warmup_ids_disposed':cache.warmup_ids is None if start>=16 else cache.warmup_ids.shape[1]==start,
           'passed':torch.equal(reference,plain) and all(v['passed'] for v in matched) and s['max_abs']<=.04 and cache.length==ids.shape[1]}
    result['cases'].append(entry);save();print(json.dumps(entry),flush=True);cache.reset()
result['passed']=all(x['passed'] for x in result['cases']);save()
raise SystemExit(0 if result['passed'] else 1)
