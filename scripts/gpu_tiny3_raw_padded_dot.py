"""Test raw-layout Core ML q·k at graph's actual Q32×K2048 geometry."""
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import coremltools as ct
import numpy as np
import torch
from torch import nn
from coreml_port.cleanup_guard import install
install();torch.set_num_threads(4)
OUT=ROOT/'evidence/04d-gpu/tiny3-raw-padded-dot';ART=ROOT/'artifacts/04d-gpu/tiny3-raw-padded-dot'
OUT.mkdir(parents=True,exist_ok=False);ART.mkdir(parents=True,exist_ok=False)
capture=np.load(ROOT/'evidence/04d-gpu/tiny3-score-candidates/arrays.npz')
q3=torch.from_numpy(capture['q']);k3=torch.from_numpy(capture['k']);v3=torch.from_numpy(capture['v'])
q=torch.cat((q3,torch.zeros(1,29,10,128)),dim=1)
k=torch.cat((k3,torch.zeros(1,2045,10,128)),dim=1)
native_score=torch.from_numpy(capture['native_score'])
native_round=torch.from_numpy(capture['native_rounded'])

class RawEinsum(nn.Module):
    def forward(self,q,k):return torch.einsum('bthd,bshd->bhts',q,k)
class RawMatmul(nn.Module):
    def forward(self,q,k):return q.permute(0,2,1,3)@k.permute(0,2,3,1)

def check(actual,reference):
    a=actual.float().numpy() if isinstance(actual,torch.Tensor) else actual
    b=reference.float().numpy() if isinstance(reference,torch.Tensor) else reference
    diff=np.abs(a.astype(np.float64)-b.astype(np.float64));ix=np.argwhere(diff!=0)
    return {'different_elements':int(len(ix)),'max_abs':float(diff.max()),
            'first_differences':[{'index':i.tolist(),'actual':float(a[tuple(i)]),'expected':float(b[tuple(i)])} for i in ix[:8]]}

mask=torch.arange(3)[None,:]<=torch.arange(3)[:,None]
report={'compute_units':'CPU_ONLY','input_capture_sha256':hashlib.sha256((ROOT/'evidence/04d-gpu/tiny3-score-candidates/arrays.npz').read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'q_variants':[list(q3.shape),list(q.shape)],'k_shape':list(k.shape),'experiments':[]}
arrays={'native_score':native_score.numpy(),'native_rounded':native_round.numpy()}
for q_label, query in [('q3',q3),('q32',q)]:
    for op_label,module in [('raw_einsum',RawEinsum()),('raw_matmul',RawMatmul())]:
        name=q_label+'_'+op_label
        module.eval()
        with torch.inference_mode(): torch_dot=module(query,k)
        traced=torch.jit.trace(module,(query,k),check_trace=False)
        converted=ct.convert(traced,inputs=[ct.TensorType(name='q',shape=query.shape,dtype=np.float32),
                                            ct.TensorType(name='k',shape=k.shape,dtype=np.float32)],
                             outputs=[ct.TensorType(name='dot',dtype=np.float32)],
                             compute_precision=ct.precision.FLOAT32,minimum_deployment_target=ct.target.macOS15,skip_model_load=True)
        package=ART/(name+'.mlpackage');converted.save(str(package))
        cm=ct.models.MLModel(str(package),compute_units=ct.ComputeUnit.CPU_ONLY)
        got=torch.from_numpy(cm.predict({'q':query.numpy(),'k':k.numpy()})['dot'])
        score=got[:,:,:3,:3]*.1
        weights=score.masked_fill(~mask[None,None],float('-inf')).softmax(-1)
        raw=torch.einsum('bhts,bshd->bthd',weights,v3)
        rounded=raw.to(torch.bfloat16)
        row={'name':name,'q_shape':list(query.shape),'torch_dot_vs_native':check(torch_dot[:,:,:3,:3]*.1,native_score),
             'coreml_scores_vs_native':check(score,native_score),
             'rounded_attention_vs_native':check(rounded,native_round),
             'problem_element':{'raw':float(raw[0,2,6,80]),'rounded':float(rounded[0,2,6,80])},
             'package_files_sha256':{str(p.relative_to(package)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(package.rglob('*')) if p.is_file()}}
        report['experiments'].append(row)
        arrays[name+'_score']=score.numpy();arrays[name+'_weights']=weights.numpy();arrays[name+'_raw']=raw.numpy();arrays[name+'_rounded']=rounded.float().numpy()
        print(name,row['coreml_scores_vs_native']['different_elements'],row['rounded_attention_vs_native']['different_elements'],row['problem_element'],flush=True)
        (OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(OUT/'arrays.npz',**arrays)
