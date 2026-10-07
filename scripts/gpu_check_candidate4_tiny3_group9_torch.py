"""Check whether candidate-4 Torch group 9 matches native on the tiny-3 capture."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
import coreml_port.graphs as graphs
import coreml_port.ordered_attention as attention
from coreml_port import dynamic_queries
from coreml_port.bf16_gelu import BF16Gelu
from coreml_port.ordered_linear import linear
from coreml_port.ordered_norm import NormWrapped
from coreml_port.ordered_xsa import ordered_xsa_dot,ordered_xsa_normalize
from coreml_port.source_exp import exp
from portable.model import load_model
torch.set_num_threads(4)
graphs.linear=linear;graphs.softmax=attention.softmax;attention.exponential=exp
graphs.xsa_normalize=ordered_xsa_normalize;graphs.xsa_dot=ordered_xsa_dot
graphs.gelu_factory=BF16Gelu
graphs.prepare_queries=dynamic_queries.prepare
graphs.attention_scores=dynamic_queries.scores
graphs.attention_product=dynamic_queries.product
BASE=ROOT/'evidence/04d-gpu/candidate4-tiny3-firstgroup'
saved=dict(np.load(BASE/'first-failure-inputs.npz'))
names=('x','ve','positions','tap0','tap12','tap23','norm_divisor')
model=load_model('cpu')
group=NormWrapped(graphs.Group(model,36).eval()).eval()
with torch.inference_mode():got,_=group(*(torch.from_numpy(saved[name]) for name in names))
actual=got.numpy();expected=saved['expected'];bad=np.argwhere(actual!=expected)
row={'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'input_sha256':hashlib.sha256((BASE/'first-failure-inputs.npz').read_bytes()).hexdigest(),
 'different_elements':int(len(bad)),
 'max_abs':float(np.max(np.abs(actual.astype(np.float64)-expected.astype(np.float64)))),
 'first_differences':[{'index':i.tolist(),'torch_candidate':float(actual[tuple(i)]),
                       'native':float(expected[tuple(i)])} for i in bad[:8]]}
(BASE/'torch-group9.json').write_text(json.dumps(row,indent=2)+'\n')
print(row,flush=True)
