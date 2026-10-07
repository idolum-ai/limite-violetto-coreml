"""Small equivalent-product layout probes on the saved layer-34 operands."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch,coremltools as ct
from torch import nn
from coreml_port.cleanup_guard import install
import coreml_port.graphs as graphs
install();torch.set_num_threads(4)
REF=ROOT/'evidence/04d-gpu/long-layer34/reference.npz'
OUT=ROOT/'evidence/04d-gpu/long-layer34/products';ART=ROOT/'artifacts/04d-gpu/long-layer34/products'
OUT.mkdir(parents=True,exist_ok=False);ART.mkdir(parents=True,exist_ok=False)
p=dict(np.load(REF));target=p['rounded'][0,9,1,26].item();full=p['rounded']
class Product(nn.Module):
    def __init__(self,mode):super().__init__();self.mode=mode
    def forward(self,w,v):
        if self.mode=='matmul':raw=w@v
        elif self.mode=='einsum':raw=torch.einsum('bhts,bhsd->bhtd',w,v)
        elif self.mode=='per_head':raw=torch.cat([w[:,i]@v[:,i] for i in range(10)],dim=1).reshape(1,10,w.shape[2],128)
        elif self.mode=='split1024':raw=(w[...,:1024]@v[...,:1024,:])+(w[...,1024:]@v[...,1024:,:])
        elif self.mode=='split256':
            raw=w[...,:256]@v[...,:256,:]
            for j in range(256,2048,256):raw=raw+(w[...,j:j+256]@v[...,j:j+256,:])
        return raw,graphs.Q(raw)
report={'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'reference_sha256':hashlib.sha256(REF.read_bytes()).hexdigest(),'target_reference':target,'rows':[]}
def save():(OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
save()
for name,mode,rows,keys in [('rows2','matmul',2,2048),('keys1026','matmul',32,1026),('einsum','einsum',32,2048),('per_head','per_head',32,2048),('split1024','split1024',32,2048),('split256','split256',32,2048)]:
    a=p['weights'][:,:,:rows,:keys].copy();b=p['val'][:,:,:keys,:].copy()
    vals=(torch.from_numpy(a),torch.from_numpy(b));module=Product(mode).eval()
    with torch.inference_mode():tor_raw,tor_q=module(*vals)
    expected=full[:,:,:rows,:]
    trace=torch.jit.trace(module,vals,check_trace=False)
    cm=ct.convert(trace,inputs=[ct.TensorType(name='w',shape=a.shape,dtype=np.float32),ct.TensorType(name='v',shape=b.shape,dtype=np.float32)],
        outputs=[ct.TensorType(name='raw',dtype=np.float32),ct.TensorType(name='rounded',dtype=np.float32)],
        compute_precision=ct.precision.FLOAT32,minimum_deployment_target=ct.target.macOS15,skip_model_load=True)
    package=ART/f'{name}.mlpackage';cm.save(str(package));host=ct.models.MLModel(str(package),compute_units=ct.ComputeUnit.CPU_ONLY)
    got=host.predict({'w':a,'v':b})
    row={'name':name,'shape_w':list(a.shape),'shape_v':list(b.shape),
         'torch_vs_full_reference':int(np.count_nonzero(tor_q.numpy()!=expected)),
         'coreml_vs_full_reference':int(np.count_nonzero(got['rounded']!=expected)),
         'coreml_vs_local_torch':int(np.count_nonzero(got['rounded']!=tor_q.numpy())),
         'target_coreml_raw':float(got['raw'][0,9,1,26]),'target_coreml_q':float(got['rounded'][0,9,1,26]),
         'target_torch_raw':float(tor_raw[0,9,1,26]),'target_torch_q':float(tor_q[0,9,1,26])}
    report['rows'].append(row);save();print(row,flush=True)
