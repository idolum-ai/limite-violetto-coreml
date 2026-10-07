"""Persist the exact tiny-3 group-8 boundary for package-only replay."""
import gc,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
import coreml_port.graphs as graphs
import coreml_port.ordered_attention as attention
from coreml_port.bf16_gelu import BF16Gelu
from coreml_port.ordered_linear import linear
from coreml_port.ordered_norm import NormWrapped
from coreml_port.ordered_xsa import ordered_xsa_dot,ordered_xsa_normalize
from coreml_port.source_exp import exp
from portable.model import load_model
from scripts.gpu_dynamic_productrows_attention import finish
import types
torch.set_num_threads(4);graphs.linear=linear;graphs.softmax=attention.softmax;attention.exponential=exp
graphs.xsa_normalize=ordered_xsa_normalize;graphs.xsa_dot=ordered_xsa_dot;graphs.gelu_factory=BF16Gelu
OUT=ROOT/'evidence/04d-gpu/tiny3-productrows-group8';OUT.mkdir(parents=True,exist_ok=True)
ids=[87555,130664,114854];divisor=torch.tensor([1280.]);model=load_model('cpu');ref={}
handle=model.backbone.layers[35].register_forward_hook(lambda m,a,o:ref.update(y=o.detach().float().clone()))
with torch.inference_mode():model.reference_hidden(torch.tensor([ids],dtype=torch.long))
handle.remove()
with torch.inference_mode():
    x,ve=NormWrapped(graphs.Embedding(model).eval())(torch.tensor([ids],dtype=torch.int32),divisor)
    tap0=x.clone();tap12=x.clone();tap23=x.clone();positions=torch.arange(3,dtype=torch.int32)
    for index in range(8):
        group=NormWrapped(graphs.Group(model,4*index).eval()).eval()
        x,tap23=group(x,ve,positions,tap0,tap12,tap23,divisor)
        if index==2:tap12=x
        del group;gc.collect()
    prefix=(x.clone(),ve.clone(),positions.clone(),tap0.clone(),tap12.clone(),tap23.clone(),divisor.clone())
    group8=graphs.Group(model,32).eval()
    for layer in group8.layers:layer.finish=types.MethodType(finish,layer)
    candidate_y,_=NormWrapped(group8)(*prefix)
arrays={name:value.numpy() for name,value in zip(('x','ve','positions','tap0','tap12','tap23','norm_divisor'),prefix)}
arrays['expected_y']=ref['y'].numpy();arrays['expected_tap23_out']=tap23.numpy()
path=OUT/'fixture.npz';np.savez_compressed(path,**arrays)
report={'ids':ids,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'fixture_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
 'shapes':{k:list(v.shape) for k,v in arrays.items()},
 'torch_candidate_different_elements':int(torch.count_nonzero(candidate_y!=ref['y'])),
 'expected_origin':'native Torch layer35 forward hook; exact layer31 graph prefix'}
(OUT/'fixture.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)
