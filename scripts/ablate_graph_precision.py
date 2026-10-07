"""Whole graph Torch ablations; retain native CPU source as an untouched oracle."""
import argparse,gc,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
import torch.nn.functional as F
import coreml_port.graphs as graphs
from portable.model import load_model
from scripts.probe_full_model import metrics
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['fp32','native_bf16','fp64','kahan32','ordered'],required=True);p.add_argument('--softmax',choices=['native','ordered'],default='native');p.add_argument('--evidence-dir',default='evidence/04c-precision');a=p.parse_args();torch.set_num_threads(4)
if a.softmax=='ordered':
    from coreml_port.ordered_attention import softmax
    graphs.softmax=softmax
m=load_model();ids=torch.tensor([json.loads((ROOT/'evidence/01-reference/local-input-validation.json').read_text())['cases'][0]['input_ids'][:32]])
original=F.linear
with torch.inference_mode():
    reference={};hidden=m(ids,capture=lambda n,v:reference.update({n:v.clone()}));scores=m.scores(hidden[:,-1:])
    def linear(x,w):
        if a.mode=='ordered':
            from coreml_port.ordered_linear import linear as ordered
            return ordered(x,w)
        if a.mode=='native_bf16':return original(x.bfloat16(),w.bfloat16()).float()
        if a.mode=='fp64':return original(x.double(),w.double()).float()
        if a.mode=='fp32':return original(x,w)
        total=x.new_zeros(x.shape[:-1]+(w.shape[0],));correction=torch.zeros_like(total)
        for start in range(0,w.shape[1],32):
            part=original(x[...,start:start+32],w[:,start:start+32]);y=part-correction;t=total+y;correction=(t-total)-y;total=t
        return total
    graphs.linear=linear;begin=time.time();x,ve=graphs.Embedding(m)(ids);tap0=x.clone();tap12=x.clone();tap23=x.clone();r={'mode':a.mode,'comparisons':{'embedding_norm':metrics(x,reference['embedding_norm'])}}
    for start in range(0,48,4):
        g=graphs.Group(m,start).eval();x,tap23=g(x,ve,torch.arange(ids.shape[1],dtype=torch.int32),tap0,tap12,tap23)
        if start==8:tap12=x.clone()
        r['comparisons'][f'layer_{start+3}']=metrics(x,reference['final_hidden' if start==44 else f'layer_{start+3}']);del g;gc.collect()
    got=graphs.Head(m)(x[:,-1:]);r['comparisons']['scores']=metrics(got,scores);r['elapsed_seconds']=time.time()-begin;r['score_pass']=r['comparisons']['scores']['max_abs']<=.04
    r['hidden_pass']=r['comparisons']['layer_47']['relative_rmse']<=.01 and r['comparisons']['layer_47']['cosine']>=.995
r['softmax']=a.softmax
(ROOT/a.evidence_dir/f'graph-{a.mode}.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
