"""Experimental Core ML pipeline. Numerical fidelity is NOT qualified."""
from dataclasses import dataclass,field
from pathlib import Path
import time
import numpy as np
import coremltools as ct
ROOT=Path(__file__).resolve().parents[1]

@dataclass
class Request:
    owner:object
    states:list=field(default_factory=list)
    length:int=0
    valid:bool=True
    closed:bool=False
    def close(self):self.states.clear();self.valid=False;self.closed=True

class Pipeline:
    capacity=2048
    valid_vocab=151667
    eos=151645
    def __init__(self,compute='CPU_ONLY',model_dir=None,identity_manifest=None):
        if (model_dir is None)!=(identity_manifest is None):
            raise ValueError('An explicit candidate requires both model_dir and identity_manifest.')
        self.model_dir=Path(model_dir).expanduser() if model_dir is not None else ROOT/'artifacts/04-coreml/model'
        self.package_identity=None
        if identity_manifest is not None:
            from .package_identity import verify_candidate_identity
            self.package_identity=verify_candidate_identity(self.model_dir,identity_manifest)
        self.model_dir=self.model_dir.resolve()
        from .cleanup_guard import install
        self.cleanup_guard=install()
        from .prediction_guard import install as install_prediction_guard
        self.prediction_guard=install_prediction_guard()
        self.owner=object();self.compute=compute;self.load_seconds={};self.models={};self.norm_divisor_inputs=set()
        for name in ['embedding',*map(str,range(12)),'head']:
            t=time.perf_counter();self.models[name]=ct.models.MLModel(str(self.model_dir/f'{name}.mlpackage'),compute_units=getattr(ct.ComputeUnit,compute));self.load_seconds[name]=time.perf_counter()-t
            if any(v.name=='norm_divisor' for v in self.models[name].get_spec().description.input):self.norm_divisor_inputs.add(name)
    def predict(self,name,inputs,state=None):
        if name in self.norm_divisor_inputs:inputs={**inputs,'norm_divisor':np.array([1280.],dtype=np.float32)}
        return self.models[name].predict(inputs,**({'state':state} if state is not None else {}))
    def request(self):return Request(self.owner,[self.models[str(i)].make_state() for i in range(12)])
    def validate(self,ids,request):
        if request.owner is not self.owner:raise ValueError('Request belongs to another pipeline.')
        if request.closed or not request.valid:raise ValueError('Request closed or invalid.')
        if not isinstance(ids,list) or not ids or any(type(i) is not int or i<0 or i>=self.valid_vocab for i in ids):raise ValueError('Expected nonempty valid integer token IDs.')
        if request.length+len(ids)>self.capacity:raise ValueError('Context exceeds 2048; no truncation.')
    def forward(self,ids,request,capture=None):
        self.validate(ids,request)
        try:
            for offset in range(0,len(ids),32):
                chunk=ids[offset:offset+32];count=len(chunk);start=request.length
                emb=self.predict('embedding',{'ids':np.array([chunk],dtype=np.int32)});x=emb['x'];ve=emb['ve'];tap0=x;tap12=x;tap23=x
                positions=np.arange(start,start+count,dtype=np.int32)
                if capture:capture('embedding_norm',x)
                for i in range(12):
                    out=self.predict(str(i),{'x':x,'ve':ve,'positions':positions,'tap0':tap0,'tap12':tap12,'tap23':tap23},state=request.states[i]);x=out['y'];tap23=out['tap23_out']
                    if i==2:tap12=x
                    if capture:capture('final_hidden' if i==11 else f'layer_{i*4+3}',x)
                request.length+=count
            scores=self.predict('head',{'x':np.ascontiguousarray(x[:,-1:])})['scores']
            if not np.isfinite(scores).all():raise FloatingPointError('Nonfinite Core ML scores.')
            return scores
        except BaseException:
            request.valid=False
            raise
