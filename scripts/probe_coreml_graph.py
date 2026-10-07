"""Compare complete graph-friendly Torch arithmetic with the qualified source."""
import json,sys,time,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from portable.model import load_model
from coreml_port.graphs import Embedding,Head,Group
from scripts.probe_full_model import metrics

torch.set_num_threads(4);m=load_model();cases=json.loads((ROOT/'evidence/01-reference/local-input-validation.json').read_text())['cases'];print(cases[0].keys(),flush=True)
ids=torch.tensor([cases[0]['input_ids'][:32]],dtype=torch.long)
ref={}
with torch.inference_mode():
    hidden=m(ids,capture=lambda n,v:ref.update({n:v.clone()}));scores=m.scores(hidden[:,-1:]);x,ve=Embedding(m)(ids);tap0=x.clone();tap12=x.clone();tap23=x.clone();results={'embedding_norm':metrics(x,ref['embedding_norm'])}
    for start in range(0,48,4):
        group=Group(m,start).eval();x,tap23=group(x,ve,torch.arange(ids.shape[1],dtype=torch.int32),tap0,tap12,tap23)
        if start==8:tap12=x.clone()
        results[f'layer_{start+3}']=metrics(x,ref['final_hidden' if start==44 else f'layer_{start+3}']);print(start,results[f'layer_{start+3}'],flush=True);del group;gc.collect()
    results['scores']=metrics(Head(m)(x[:,-1:]),scores)
(ROOT/'evidence/04-coreml/torch-graph.json').write_text(json.dumps({'tokens':ids.tolist(),'comparisons':results},indent=2)+'\n')
