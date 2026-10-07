"""Fresh-input Core ML validation of the experimental nonpositive exponential."""
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np, torch, coremltools as ct
from torch import nn
from coreml_port.source_exp import exp
from coreml_port.cleanup_guard import install
install();torch.set_num_threads(4)
p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);a=p.parse_args()
out=ROOT/'evidence/04d-cpu'/a.attempt;art=ROOT/'artifacts/04d-cpu'/a.attempt
out.mkdir(parents=True,exist_ok=False);art.mkdir(parents=True,exist_ok=False)
class Graph(nn.Module):
    def forward(self,x):return exp(x)
shape=(1,16384);graph=Graph().eval()
trace=torch.jit.trace(graph,torch.zeros(shape),check_trace=False)
model=ct.convert(trace,inputs=[ct.TensorType(name='x',shape=shape,dtype=np.float32)],outputs=[ct.TensorType(name='y',dtype=np.float32)],compute_precision=ct.precision.FLOAT32,minimum_deployment_target=ct.target.macOS15,skip_model_load=True)
path=art/'exp.mlpackage';model.save(str(path));cm=ct.models.MLModel(str(path),compute_units=ct.ComputeUnit.CPU_ONLY)
rng=np.random.default_rng(72922026)
cases={'uniform':-rng.uniform(0,104,262144).astype(np.float32),
       'log_uniform':-np.exp(rng.uniform(-90,np.log(104),262144)).astype(np.float32)}
centers=np.array([-k*np.log(2) for k in range(151)],dtype=np.float32)
near=np.concatenate([centers]+[np.nextafter(centers,np.float32(direction)) for direction in [-np.inf,np.inf]])
cases['boundaries']=np.resize(np.concatenate([near,np.array([0,-0.,-104,-105,-np.inf],np.float32)]),16384)
report={'trace_input':'all zeros; every test input is supplied after conversion','compute':'CPU_ONLY','cases':{}}
for name,values in cases.items():
    rows=[];failures=[]
    for i in range(0,len(values),16384):
        x=values[i:i+16384].reshape(shape);tx=torch.from_numpy(x)
        with torch.inference_mode():want=tx.exp().numpy();source=graph(tx).numpy()
        got=cm.predict({'x':x})['y'];bad=got!=want;sbad=source!=want
        rows.append({'elements':x.size,'coreml_different':int(bad.sum()),'torch_different':int(sbad.sum()),'max_abs':float(np.abs(got-want).max())})
        mask=bad|sbad
        if mask.any():failures.append(np.stack([x[mask],want[mask],source[mask],got[mask]],axis=1))
    report['cases'][name]={'batches':rows,'different':sum(r['coreml_different'] for r in rows),'torch_different':sum(r['torch_different'] for r in rows)}
    if failures:np.save(out/(name+'-failures.npy'),np.concatenate(failures))
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(name,report['cases'][name]['different'],report['cases'][name]['torch_different'],flush=True)
