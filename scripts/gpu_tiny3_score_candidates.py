"""Test alternate Core ML q·k layouts and FP64 oracle at BF16 attention output."""
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
OUT=ROOT/'evidence/04d-gpu/tiny3-score-candidates';OUT.mkdir(parents=True,exist_ok=False)
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
q=saved['q'].float();k=saved['k'].repeat_interleave(5,dim=2).float();v=saved['v'].repeat_interleave(5,dim=2).float()
native_dot=torch.einsum('bthd,bshd->bhts',q,k);native_score=native_dot*.1
mask=torch.arange(3)[None,:]<=torch.arange(3)[:,None]
if attn.window>=0:mask=mask & (torch.arange(3)[None,:]>=torch.arange(3)[:,None]-attn.window)
def attention_result(scores):
    weights=scores.masked_fill(~mask[None,None],float('-inf')).softmax(-1)
    raw=torch.einsum('bhts,bshd->bthd',weights,v)
    return weights,raw,raw.to(torch.bfloat16)
with torch.inference_mode(): native_weights,native_raw,native_rounded=attention_result(native_score)

def metric(actual,reference):
    a=actual.float().numpy() if isinstance(actual,torch.Tensor) else actual
    b=reference.float().numpy() if isinstance(reference,torch.Tensor) else reference
    d=np.abs(a.astype(np.float64)-b.astype(np.float64));ix=np.argwhere(d!=0)
    return {'different_elements':int(len(ix)),'max_abs':float(d.max()),
            'first_differences':[{'index':i.tolist(),'actual':float(a[tuple(i)]),'expected':float(b[tuple(i)])} for i in ix[:8]]}

report={'ids':ids,'compute_units':'CPU_ONLY','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            ROOT/'portable/model.py',ROOT/'sources/value-model/modeling_backbone.py')},
        'reference':{'problem_dot':native_dot[0,6,2,:].tolist(),'problem_score':native_score[0,6,2,:].tolist(),
                     'problem_raw':float(native_raw[0,2,6,80]),'problem_rounded':float(native_rounded[0,2,6,80])},
        'experiments':[]}
arrays={'native_dot':native_dot.numpy(),'native_score':native_score.numpy(),'native_weights':native_weights.numpy(),
        'native_raw':native_raw.numpy(),'native_rounded':native_rounded.float().numpy(),
        'q':q.numpy(),'k':k.numpy(),'v':v.numpy()}
def add(name,dot,score,package=None):
    with torch.inference_mode(): weights,raw,rounded=attention_result(torch.from_numpy(score))
    row={'name':name,'dot':metric(dot,native_dot),'scores':metric(score,native_score),
         'weights':metric(weights,native_weights),'raw':metric(raw,native_raw),
         'rounded':metric(rounded,native_rounded),
         'problem_element':{'score':score[0,6,2,:].tolist(),'raw':float(raw[0,2,6,80]),
                            'rounded':float(rounded[0,2,6,80])}}
    if package is not None:
        row['package']=str(package.relative_to(ROOT))
        row['package_files_sha256']={str(p.relative_to(package)):hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in sorted(package.rglob('*')) if p.is_file()}
    report['experiments'].append(row)
    arrays[name+'_dot']=dot
    arrays[name+'_score']=score
    arrays[name+'_weights']=weights.numpy()
    arrays[name+'_raw']=raw.numpy()
    arrays[name+'_rounded']=rounded.float().numpy()
    print(name,row['scores']['different_elements'],row['rounded']['different_elements'],row['problem_element'],flush=True)
    (OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(OUT/'arrays.npz',**arrays)

for name in ('raw_einsum','raw_matmul','per_head','reversed'):
    package=ROOT/'artifacts/04d-gpu/tiny3-dot-layouts'/f'{name}.mlpackage'
    cm=ct.models.MLModel(str(package),compute_units=ct.ComputeUnit.CPU_ONLY)
    dot=cm.predict({'q':q.numpy(),'k':k.numpy()})['dot']
    add(name,dot,np.float32(dot*np.float32(.1)),package)

q64=q.numpy().astype(np.float64)[0].transpose(1,0,2)
k64=k.numpy().astype(np.float64)[0].transpose(1,0,2)
oracle64=(q64[:,:,None,:]*k64[:,None,:,:]).sum(axis=-1,dtype=np.float64)[None]
oracle_dot=oracle64.astype(np.float32)
add('fp64_sum_then_fp32_dot_and_scale',oracle_dot,np.float32(oracle_dot*np.float32(.1)))
add('fp64_sum_then_fp32_scale_before_cast',oracle_dot,np.float32(oracle64*float(np.float32(.1))))
add('fp64_sum_then_decimal_scale_before_cast',oracle_dot,np.float32(oracle64*.1))
