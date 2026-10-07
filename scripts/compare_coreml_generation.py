"""Same-input/seed sampled generation comparison to the qualified CPU model."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from portable.model import load_model
from portable.generate import generate,tokenizer

torch.set_num_threads(4);candidate=json.loads((ROOT/'evidence/04-coreml/generation-cpu.json').read_text());m=load_model();tok=tokenizer()
reference=generate(m,candidate['prompt_ids'],max_new_tokens=1024,greedy=False,seed=0,additional_stop_ids=(151643,));reference['text']=tok.decode(reference['generated_ids'],skip_special_tokens=True)
a=candidate['generated_ids'];b=reference['generated_ids'];common=0
for x,y in zip(a,b):
    if x!=y:break
    common+=1
result={'reference':reference,'candidate_record':'generation-cpu.json','identical_inputs_and_sampling_settings':True,'identical_token_sequence':a==b,'common_generated_prefix_tokens':common,'first_divergence':None if a==b else {'index':common,'candidate':a[common] if common<len(a) else None,'reference':b[common] if common<len(b) else None},'candidate_boxed_391_after_think':'\\boxed{391}' in candidate['text'].split('</think>')[-1],'reference_boxed_391_after_think':'\\boxed{391}' in reference['text'].split('</think>')[-1]}
(ROOT/'evidence/04-coreml/generation-comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='reference'},indent=2))
