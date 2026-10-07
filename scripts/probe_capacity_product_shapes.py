"""Controlled value-product geometry probes on the saved capacity failure."""
import hashlib, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np, torch, coremltools as ct
from torch import nn
from coreml_port.graphs import Q
from coreml_port.prediction_guard import install
install();torch.set_num_threads(4)
base=ROOT/'evidence/04d-cpu/capacity-layer0'
out=base/'product-shapes';art=ROOT/'artifacts/04d-cpu/capacity-layer0/product-shapes'
out.mkdir(parents=True,exist_ok=False);art.mkdir(parents=True,exist_ok=False)
saved=dict(np.load(base/'reference.npz'))
w=torch.from_numpy(saved['weights']);v=torch.from_numpy(saved['val'])
targets=[(0,9,8,83),(0,2,30,121)]
def compare(x,y):
    return {'different_elements':int(np.count_nonzero(x!=y)),
            'max_abs':float(np.max(np.abs(x.astype(np.float64)-y.astype(np.float64)))),
            'targets':[{'index':list(i),'actual':float(x[i]),'reference':float(y[i])} for i in targets]}
report={'status':'starting','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'input_sha256':hashlib.sha256((base/'reference.npz').read_bytes()).hexdigest(),'variants':{}}
def save():(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
with torch.inference_mode():
    precise=(w.double()@v.double()).float()
report['higher_precision_control']={'raw':compare(precise.numpy(),saved['raw']),
                                    'rounded':compare(Q(precise).numpy(),saved['rounded'])}
save()
class Product(nn.Module):
    def __init__(self,rows=None,active_keys=None,key_chunk=None):
        super().__init__();self.rows=rows;self.active_keys=active_keys;self.key_chunk=key_chunk
    def forward(self,weights,values):
        if self.rows:
            raw=torch.cat([weights[:,:,i:i+self.rows]@values for i in range(0,32,self.rows)],dim=2)
        elif self.active_keys:
            raw=weights[:,:,:,:self.active_keys]@values[:,:,:self.active_keys]
        elif self.key_chunk:
            raw=weights[:,:,:,:self.key_chunk]@values[:,:,:self.key_chunk]
            for i in range(self.key_chunk,2048,self.key_chunk):
                raw=raw+(weights[:,:,:,i:i+self.key_chunk]@values[:,:,i:i+self.key_chunk])
        else:raw=weights@values
        return raw,Q(raw)
variants={**{f'rows{n}':Product(rows=n) for n in (1,2,4,8,16)},
          'active1090':Product(active_keys=1090),'keychunks256':Product(key_chunk=256)}
arrays={}
for name,module in variants.items():
    with torch.inference_mode():expected=module(w,v)
    row={'torch_rounded':compare(expected[1].numpy(),saved['rounded'])}
    trace=torch.jit.trace(module,(w,v),check_trace=False)
    converted=ct.convert(trace,inputs=[ct.TensorType(name='weights',shape=w.shape,dtype=np.float32),ct.TensorType(name='values',shape=v.shape,dtype=np.float32)],outputs=[ct.TensorType(name='raw',dtype=np.float32),ct.TensorType(name='rounded',dtype=np.float32)],compute_precision=ct.precision.FLOAT32,minimum_deployment_target=ct.target.macOS15,skip_model_load=True)
    package=art/f'{name}.mlpackage';converted.save(str(package))
    cm=ct.models.MLModel(str(package),compute_units=ct.ComputeUnit.CPU_ONLY)
    got=cm.predict({'weights':saved['weights'],'values':saved['val']})
    row.update(coreml_raw=compare(got['raw'],saved['raw']),coreml_rounded=compare(got['rounded'],saved['rounded']))
    report['variants'][name]=row;save();arrays[name+'_raw']=got['raw'];arrays[name+'_rounded']=got['rounded']
    print(name,row['coreml_rounded'],flush=True)
report['status']='complete';save();np.savez_compressed(out/'outputs.npz',**arrays)
