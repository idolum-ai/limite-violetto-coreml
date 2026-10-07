"""Run the pinned original head method and exercise generation-loop stop paths."""
import ast,json,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
import torch.nn.functional as F
from portable.model import load_model
from portable.generate import generate
torch.set_num_threads(4)
source=ROOT/'sources/runtime/paradigma-inc-limite-violetto-cd68d27/src/limite_vllm/modeling.py'
tree=ast.parse(source.read_text())
method=next(node for cls in tree.body if isinstance(cls,ast.ClassDef) for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='compute_logits')
scope={'torch':torch,'F':F};exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),scope)
m=load_model('cpu');ids=json.loads((ROOT/'evidence/01-reference/local-input-validation.json').read_text())['cases'][0]['input_ids']
h=m.reference_hidden(torch.tensor([ids]))[:,-1:]
c=m.config.softcap_logits
holder=SimpleNamespace(lm_head=SimpleNamespace(weight=m.backbone.embed_tokens.weight),head_precision_mode=m.config.lm_head_precision_mode,softcap_a=c['a'],softcap_b=c['b'],softcap_c=c['c'])
original=scope['compute_logits'](holder,h);adapted=m.scores(h)
result={'head':{'scope':'Original pinned compute_logits method run on CPU with real hidden state and tied actor weights; not original CUDA serving execution','max_abs':float((original-adapted).abs().max()),'exact_match':torch.equal(original,adapted)}}
# A controlled backend tests loop termination and cleanup independently of what
# the language model chooses to emit. The real model is tested separately.
class Backend:
    device=torch.device('cpu');capacity=16
    config=SimpleNamespace(eos_token_id=151645,tokenizer_vocab_size=151667)
    def __init__(self,tokens):self.tokens=iter(tokens);self.calls=0;self.resets=0
    def validate_ids(self,ids):pass
    def new_cache(self):return SimpleNamespace(reset=self.reset)
    def reset(self):self.resets+=1
    def __call__(self,ids,cache):self.calls+=1;return torch.zeros(1,1,1)
    def scores(self,hidden):
        scores=torch.zeros(151680);scores[next(self.tokens)]=100;return scores
eos=Backend([151645]);er=generate(eos,[1],8)
length=Backend([151643,2]);lr=generate(length,[1],2)
cancel=Backend([1])
def interrupt(token):raise KeyboardInterrupt('Controlled test interruption')
try:generate(cancel,[1],8,on_token=interrupt)
except KeyboardInterrupt:pass
result['loop_checks']={'eos_stops_without_extra_forward':er['finish_reason']=='eos' and er['complete'] and eos.calls==1 and eos.resets==1,'tokenizer_eos_does_not_stop_model':lr['generated_ids']==[151643,2] and lr['finish_reason']=='length' and not lr['complete'] and length.resets==1,'callback_interruption_disposes_state':cancel.calls==1 and cancel.resets==1}
extra=Backend([151643,2]);xr=generate(extra,[1],2,additional_stop_ids=[151643])
result['loop_checks']['explicit_extra_stop_is_recorded']=xr['finish_reason']=='stop_token' and xr['complete'] and xr['stop_token_ids']==[151643,151645] and xr['completion_policy']=='published_eos_plus_explicit_stops' and extra.calls==1 and extra.resets==1
result['passed']=result['head']['exact_match'] and all(result['loop_checks'].values())
(ROOT/'evidence/03-portable/head-and-stops.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
