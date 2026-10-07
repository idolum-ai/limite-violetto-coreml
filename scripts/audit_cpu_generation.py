"""Audit sampled real generation against matched full prefixes on CPU."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from portable.model import load_model
from portable.generate import generate,tokenizer
from probe_full_model import metrics

torch.set_num_threads(4)
out=ROOT/'evidence/03b-cache/cpu-generation.json'
if out.exists():raise FileExistsError(out)
m=load_model('cpu');tok=tokenizer()
prompt=json.loads((ROOT/'evidence/01-reference/local-input-validation.json').read_text())['cases'][0]['input_ids']
result={'device':'cpu','attention_min_query_rows':m.attention_min_query_rows,'cache_warmup_tokens':m.cache_warmup_tokens,'audits':[],
        'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['portable/model.py','portable/generate.py','scripts/audit_cpu_generation.py']}}
def save():out.write_text(json.dumps(result,indent=2)+'\n')
original_scores=m.scores;last_score=None;generated=[]
def capture_scores(hidden):
    global last_score
    last_score=original_scores(hidden).detach().clone()
    return last_score
m.scores=capture_scores
def audit(label,prefix):
    ref=m.reference_hidden(torch.tensor([prefix]));rs=original_scores(ref[:,-1:]);v=metrics(rs,last_score)
    entry={'at_generated_tokens':label,'prefix_length':len(prefix),'scores':v,'top1_same':int(rs.argmax())==int(last_score.argmax()),'passed':v['max_abs']<=.04}
    result['audits'].append(entry);save();print(json.dumps(entry),flush=True)
def on_token(token):
    if len(generated)%128==0:audit(len(generated),prompt+generated)
    generated.append(token)
sample=generate(m,prompt,1024,greedy=False,seed=0,on_token=on_token,additional_stop_ids=[tok.eos_token_id])
audit('terminal',prompt+generated[:-1])
sample['text']=tok.decode(sample['generated_ids'],skip_special_tokens=True)
sample['timing_note']='Includes full-prefix numerical audits; not a generation benchmark.'
result['generation']=sample
result['boxed_391_after_think']='</think>' in sample['text'] and '\\boxed{391}' in sample['text'].rsplit('</think>',1)[-1]
result['passed']=all(x['passed'] for x in result['audits']) and sample['complete'] and result['boxed_391_after_think']
save();print('COMPLETE',sample['complete'],'tokens',sample['generated_tokens'],'passed',result['passed'],flush=True)
raise SystemExit(0 if result['passed'] else 1)
