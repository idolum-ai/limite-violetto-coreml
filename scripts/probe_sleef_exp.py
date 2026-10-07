"""Bounded nonpositive SLEEF exp expression; diagnose Core ML FMA behavior.

Re-expresses xexpf from SLEEF 56e1f79cb140fb9326d612d0be06b5250565cade.
Copyright Naoki Shibata and contributors; retained Boost license in sources.
This is an experiment, not selected by any full candidate.
"""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np,coremltools as ct
from torch import nn
from coreml_port.cleanup_guard import install
install();torch.set_num_threads(4)
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--modes',default='double,naive,addcmul');a=p.parse_args()
out=ROOT/'evidence/04d-cpu'/a.run;art=ROOT/'artifacts/04d-cpu'/a.run;out.mkdir(parents=True,exist_ok=False);art.mkdir(parents=True,exist_ok=False)
class Exp(nn.Module):
    def __init__(self,mode):super().__init__();self.mode=mode;self.register_buffer('powers',2.**torch.arange(-80,1,dtype=torch.float32))
    def fma(self,a,b,c):
        if self.mode=='naive':return a*b+c
        # Materialize source constants as FP32 before any double arithmetic.
        a=a if isinstance(a,torch.Tensor) else torch.tensor(a,dtype=torch.float32)
        b=b if isinstance(b,torch.Tensor) else torch.tensor(b,dtype=torch.float32)
        c=c if isinstance(c,torch.Tensor) else torch.tensor(c,dtype=torch.float32)
        if self.mode=='double':return (a.double()*b.double()+c.double()).float()
        if self.mode=='dd':
            ta=a*4097.;ah=ta-(ta-a);al=a-ah
            tb=b*4097.;bh=tb-(tb-b);bl=b-bh
            product=a*b
            e1=product-ah*bh;e2=e1-al*bh;e3=e2-ah*bl;error=al*bl-e3
            total=product+c;bb=total-product
            remainder=(product-(total-bb))+(c-bb)
            return total+(error+remainder)
        return torch.addcmul(c,a,b)
    def forward(self,x):
        d=torch.clamp(x,min=-104,max=0)
        q=torch.round(d*1.4426950408889634)
        s=self.fma(q,-0.693145751953125,d)
        s=self.fma(q,-1.428606765330187e-6,s)
        u=0.00019852761761285365
        for c in [0.0013930435525253415,0.008333360776305199,0.041666485369205475,0.1666666716337204,0.5]:u=self.fma(u,s,c)
        u=1+self.fma(s*s,u,s)
        half=torch.floor(q*.5)
        u=(u*self.powers[(half+80).long()])*self.powers[(q-half+80).long()]
        return torch.where(x < -104,torch.zeros_like(x),u)
captured=torch.from_numpy(np.load(ROOT/'evidence/04d-cpu/decomposed-2/decomposed-outputs.npz')['shifted_expected']).flatten()
g=torch.Generator().manual_seed(20260922);random=-104*torch.rand(20000,generator=g)
x=torch.cat((captured,random,torch.tensor([0.,-104.,-105.,-1.,float('-inf')]))).reshape(1,-1)
want=x.exp();report={'elements':x.numel(),'torch':{},'coreml':{}}
for mode in a.modes.split(','):
    graph=Exp(mode).eval()
    with torch.inference_mode():expected=graph(x)
    report['torch'][mode]={'different_elements':int((expected!=want).sum()),'max_abs':float((expected-want).abs().max())}
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    if mode=='double':continue
    traced=torch.jit.trace(graph,x,check_trace=False)
    model=ct.convert(traced,inputs=[ct.TensorType(name='x',shape=x.shape,dtype=np.float32)],outputs=[ct.TensorType(name='y',dtype=np.float32)],compute_precision=ct.precision.FLOAT32,minimum_deployment_target=ct.target.macOS15,skip_model_load=True);path=art/(mode+'.mlpackage');model.save(str(path));cm=ct.models.MLModel(str(path),compute_units=ct.ComputeUnit.CPU_ONLY);got=cm.predict({'x':x.numpy()})['y']
    report['coreml'][mode]={'different_elements':int(np.count_nonzero(got!=want.numpy())),'max_abs':float(np.abs(got-want.numpy()).max())}
    np.savez_compressed(out/(mode+'.npz'),x=x.numpy(),expected=want.numpy(),actual=got)
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(mode,report,flush=True)
