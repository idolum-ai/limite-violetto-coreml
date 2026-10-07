"""Test whether exact-shape Core ML scores still cross tiny3 BF16 attention."""
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import coremltools as ct
import numpy as np
import torch
from coreml_port.cleanup_guard import install
from portable.model import load_model
install();torch.set_num_threads(4)
OUT=ROOT/'evidence/04d-gpu/tiny3-coreml-score-effect';OUT.mkdir(parents=True,exist_ok=False)
ids=[87555,130664,114854];model=load_model('cpu');attn=model.backbone.layers[32].self_attn
saved={};old=attn.project
def project(hidden,scale,ve,cos,sin):
    result=old(hidden,scale,ve,cos,sin)
    saved.update(q=result[0].detach().clone(),k=result[1].detach().clone(),v=result[2].detach().clone())
    return result
attn.project=project
try:
    with torch.inference_mode():model.reference_hidden(torch.tensor([ids],dtype=torch.long))
finally:attn.project=old
q=saved['q'].float().permute(0,2,1,3).contiguous()
k=saved['k'].repeat_interleave(5,dim=2).float().permute(0,2,3,1).contiguous()
v=saved['v'].repeat_interleave(5,dim=2).float().permute(0,2,1,3).contiguous()
native_scores=torch.einsum('bthd,bshd->bhts',saved['q'].float(),saved['k'].repeat_interleave(5,dim=2).float())*.1
package=ROOT/'artifacts/04d-gpu/tiny3-dot-coreml/q3-k3-front.mlpackage'
cm=ct.models.MLModel(str(package),compute_units=ct.ComputeUnit.CPU_ONLY)
got_scores=torch.from_numpy(cm.predict({'q':q.numpy(),'k':k.numpy()})['scores'])
mask=torch.arange(3)[None,:]<=torch.arange(3)[:,None]
if attn.window>=0:mask=mask & (torch.arange(3)[None,:]>=torch.arange(3)[:,None]-attn.window)
def outcome(scores):
    weights=scores.masked_fill(~mask[None,None],float('-inf')).softmax(-1)
    raw=torch.einsum('bhts,bhsd->bthd',weights,v)
    return weights,raw,raw.to(torch.bfloat16)
with torch.inference_mode():
    nw,nraw,nround=outcome(native_scores);cw,craw,cround=outcome(got_scores)
def check(a,b):
    d=(a.float()-b.float()).abs();return {'different_elements':int(torch.count_nonzero(d)),'max_abs':float(d.max())}
report={'ids':ids,'compute_units':'CPU_ONLY','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'package_files_sha256':{str(p.relative_to(package)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(package.rglob('*')) if p.is_file()},
        'scores':check(got_scores,native_scores),'weights':check(cw,nw),'raw':check(craw,nraw),'rounded':check(cround,nround),
        'problem_element':{'native_score':native_scores[0,6,2,:].tolist(),'coreml_score':got_scores[0,6,2,:].tolist(),
                           'native_weight':nw[0,6,2,:].tolist(),'coreml_weight':cw[0,6,2,:].tolist(),
                           'native_raw':float(nraw[0,2,6,80]),'coreml_raw':float(craw[0,2,6,80]),
                           'native_rounded':float(nround[0,2,6,80]),'coreml_rounded':float(cround[0,2,6,80])}}
(OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)
