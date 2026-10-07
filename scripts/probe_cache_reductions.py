"""Attention-only ablations, preserving original files, weights and thresholds."""
import argparse,hashlib,json,sys,time
from functools import partial
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
import portable.model as pm
from probe_full_model import metrics

original=pm.attend
def query_pad(q,k,v,pos,window,scale,groups):
    start=int(pos[0]);end=k.shape[1]
    if start==0:return original(q,k,v,pos,window,scale,groups)
    qp=q.new_zeros((q.shape[0],end,q.shape[2],q.shape[3]));qp[:,start:]=q
    return original(qp,k,v,torch.arange(end,device=q.device),window,scale,groups)[:,start:]

def fp64(q,k,v,pos,window,scale,groups):
    k=k.repeat_interleave(groups,dim=2);v=v.repeat_interleave(groups,dim=2)
    w=torch.einsum('bthd,bshd->bhts',q.double(),k.double())*scale
    w=w.masked_fill(~pm.attention_mask(pos,k.shape[1],window)[None,None],float('-inf')).softmax(-1)
    return torch.einsum('bhts,bshd->bthd',w,v.double()).to(v.dtype)

def minimum_rows(q,k,v,pos,window,scale,groups,rows):
    count=q.shape[1]
    if count>=rows:return original(q,k,v,pos,window,scale,groups)
    padded=q.new_zeros((q.shape[0],rows,q.shape[2],q.shape[3]));padded[:,-count:]=q
    p=torch.zeros(rows,device=q.device,dtype=pos.dtype);p[-count:]=pos
    return original(padded,k,v,p,window,scale,groups)[:,-count:]

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--attempt',required=True);p.add_argument('--long',action='store_true');p.add_argument('--modes',default='native,query_pad,fp64');args=p.parse_args()
    torch.set_num_threads(4);out=ROOT/'evidence/03b-cache'/args.attempt;out.mkdir()
    result={'device':args.device,'cases':[],'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    def save():
        (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    model=pm.load_model(args.device,attention_min_query_rows=0);data=json.loads((ROOT/'evidence/01-reference/local-input-validation.json').read_text())['cases']
    sequences=[(f'prompt_{i}',c['input_ids']) for i,c in enumerate(data)]
    if args.long:sequences.append(('window',(data[0]['input_ids']*24)[:1026]))
    modes={'native':original,'query_pad':query_pad}
    modes.update({f'rows_{n}':partial(minimum_rows,rows=n) for n in [4,8,16,32]})
    if args.device=='cpu':modes['fp64']=fp64
    modes={k:v for k,v in modes.items() if k in args.modes.split(',')}
    for name,ids in sequences:
        print(name,flush=True);x=torch.tensor([ids],device=model.device);n=len(ids)
        pm.attend=original;ref=model.reference_hidden(x);scores=model.scores(ref[:,-1:])
        for mode,fn in modes.items():
            pm.attend=fn;t=time.perf_counter();full=model(x)
            cache=model.new_cache();sizes=[256,256,256,256,1,1] if name=='window' else [n-3,1,1,1]
            chunks=[];start=0
            for size in sizes:
                chunks.append(model(x[:,start:start+size],cache));start+=size
            cached=torch.cat(chunks,dim=1)
            entry={'case':name,'mode':mode,'seconds':time.perf_counter()-t,
                   'full_vs_source_hidden':metrics(ref,full),'full_vs_source_scores':metrics(scores,model.scores(full[:,-1:])),
                   'cache_vs_source_hidden_last':metrics(ref[:,-1:],cached[:,-1:]),'cache_vs_source_scores':metrics(scores,model.scores(cached[:,-1:])),
                   'cache_vs_same_mode_hidden':metrics(full,cached),'cache_vs_same_mode_scores':metrics(model.scores(full[:,-1:]),model.scores(cached[:,-1:]))}
            result['cases'].append(entry);save();print(json.dumps(entry),flush=True);cache.reset()
    pm.attend=original
if __name__=='__main__':main()
