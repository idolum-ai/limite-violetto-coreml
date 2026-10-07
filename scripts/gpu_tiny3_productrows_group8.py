"""Uninstrumented group-8 CPU Core ML gate with exact tiny3 Torch prefix."""
import gc
import hashlib
import json
import sys
import types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import coremltools as ct
import numpy as np
import torch
import coreml_port.graphs as graphs
import coreml_port.ordered_attention as attention
from coreml_port.bf16_gelu import BF16Gelu
from coreml_port.cleanup_guard import install
from coreml_port.ordered_linear import linear
from coreml_port.ordered_norm import NormWrapped
from coreml_port.ordered_xsa import ordered_xsa_dot,ordered_xsa_normalize
from coreml_port.source_exp import exp
from portable.model import load_model
from scripts.gpu_dynamic_productrows_attention import finish as dynamic_finish
install();torch.set_num_threads(4)
graphs.linear=linear;graphs.softmax=attention.softmax;attention.exponential=exp
graphs.xsa_normalize=ordered_xsa_normalize;graphs.xsa_dot=ordered_xsa_dot;graphs.gelu_factory=BF16Gelu
OUT=ROOT/'evidence/04d-gpu/tiny3-productrows-group8';ART=ROOT/'artifacts/04d-gpu/tiny3-productrows-group8'
OUT.mkdir(parents=True,exist_ok=False);ART.mkdir(parents=True,exist_ok=False)
PACKAGE=ART/'8.mlpackage';ids=[87555,130664,114854];DIVISOR=torch.tensor([1280.])
report={'ids':ids,'compute_units':'CPU_ONLY','query_row_policy':'front-pad score rows min(32,last_absolute_position+1); weighted product actual query rows',
        'key_capacity':2048,'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            ROOT/'scripts/gpu_tiny3_productrows_group8.py',ROOT/'scripts/gpu_dynamic_productrows_attention.py',
            ROOT/'coreml_port/graphs.py',ROOT/'coreml_port/ordered_attention.py',ROOT/'coreml_port/source_exp.py',
            ROOT/'coreml_port/ordered_linear.py',ROOT/'coreml_port/ordered_norm.py',ROOT/'coreml_port/ordered_xsa.py',
            ROOT/'coreml_port/bf16_gelu.py')},'status':'starting'}
def save():(OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
def compare(actual,expected):
    a=actual.astype(np.float32) if isinstance(actual,np.ndarray) else actual.detach().float().numpy()
    b=expected.detach().float().numpy();d=np.abs(a.astype(np.float64)-b.astype(np.float64));ix=np.argwhere(d!=0)
    return {'different_elements':int(len(ix)),'max_abs':float(d.max()),
            'first_differences':[{'index':i.tolist(),'actual':float(a[tuple(i)]),'expected':float(b[tuple(i)])} for i in ix[:8]]}
save();model=load_model('cpu');ref={};handles=[]
for i in (31,35):
    def keep(module,args,output,index=i):ref[index]=output.detach().clone()
    handles.append(model.backbone.layers[i].register_forward_hook(keep))
try:
    with torch.inference_mode():model.reference_hidden(torch.tensor([ids],dtype=torch.long))
finally:
    for h in handles:h.remove()
with torch.inference_mode():
    x,ve=NormWrapped(graphs.Embedding(model).eval())(torch.tensor([ids],dtype=torch.int32),DIVISOR)
    tap0=x.clone();tap12=x.clone();tap23=x.clone();positions=torch.arange(3,dtype=torch.int32)
    for index in range(8):
        group=NormWrapped(graphs.Group(model,4*index).eval()).eval()
        x,tap23=group(x,ve,positions,tap0,tap12,tap23,DIVISOR)
        if index==2:tap12=x
        del group;gc.collect()
    report['prefix_group7_vs_reference_layer31']=compare(x,ref[31]);save()
    prefix=(x.clone(),ve.clone(),positions.clone(),tap0.clone(),tap12.clone(),tap23.clone(),DIVISOR.clone())
    group8=graphs.Group(model,32).eval()
    for layer in group8.layers:layer.finish=types.MethodType(dynamic_finish,layer)
    wrapped=NormWrapped(group8).eval()
    torch_y,torch_tap=wrapped(*prefix)
    report['torch_candidate_y_vs_reference_layer35']=compare(torch_y,ref[35]);save()
    for name,value in wrapped.named_buffers():
        if '_cache_' in name:value.zero_()
    trace=torch.jit.trace(wrapped,prefix,check_trace=False)
seq=ct.RangeDim(1,32,default=3)
def tensor(name,shape,dtype=np.float32):return ct.TensorType(name=name,shape=shape,dtype=dtype)
inputs=[tensor('x',(1,seq,1280)),tensor('ve',(1,seq,2,128)),tensor('positions',(seq,),np.int32)]
inputs += [tensor(name,(1,seq,1280)) for name in ('tap0','tap12','tap23')]
inputs += [tensor('norm_divisor',(1,))]
states=[ct.StateType(name=n,wrapped_type=ct.TensorType(shape=v.shape,dtype=np.float16))
        for n,v in wrapped.named_buffers() if '_cache_' in n]
converted=ct.convert(trace,inputs=inputs,outputs=[tensor('y',None),tensor('tap23_out',None)],states=states,
                     compute_precision=ct.precision.FLOAT32,minimum_deployment_target=ct.target.macOS15,skip_model_load=True)
converted.save(str(PACKAGE));report['status']='converted'
report['package_files_sha256']={str(p.relative_to(PACKAGE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(PACKAGE.rglob('*')) if p.is_file()};save()
payload={name:value.numpy() for name,value in zip(('x','ve','positions','tap0','tap12','tap23','norm_divisor'),prefix)}
cm=ct.models.MLModel(str(PACKAGE),compute_units=ct.ComputeUnit.CPU_ONLY)
got=cm.predict(payload,state=cm.make_state())
report['coreml_y_vs_reference_layer35']=compare(got['y'],ref[35]);report['coreml_tap23_vs_reference']=compare(got['tap23_out'],tap23)
baseline_package=ROOT/'artifacts/04d-cpu/full-candidate-3/8.mlpackage'
baseline=ct.models.MLModel(str(baseline_package),compute_units=ct.ComputeUnit.CPU_ONLY)
old=baseline.predict(payload,state=baseline.make_state())
report['baseline_candidate3_y_vs_reference_layer35']=compare(old['y'],ref[35])
report['status']='complete';save()
print('baseline',report['baseline_candidate3_y_vs_reference_layer35'],flush=True)
print('dynamic',report['coreml_y_vs_reference_layer35'],flush=True)
