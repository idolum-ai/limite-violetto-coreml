"""Retain actual stop-token emissions, including text hidden by clean decoding."""
import json
from pathlib import Path
from tokenizers import Tokenizer
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'evidence/03-portable'
tok=Tokenizer.from_file(str(ROOT/'sources/model/tokenizer.json'))
observations=[]
items=[]
for run in ['run2','run3']:
    r=json.loads((out/run/'results.json').read_text())
    items.extend((f'{run}/results.json#{i}',g) for i,g in enumerate(r['generation']))
for name in ['cli-math','cli-math-sampled','cli-math-full-budget','cli-math-explicit-stop']:
    items.append((name+'.json',json.loads((out/f'{name}.json').read_text())))
for path,g in items:
    ids=g['generated_ids'];first=ids.index(151643) if 151643 in ids else None
    prefix=tok.decode(ids[:first+1],skip_special_tokens=False) if first is not None else None
    observations.append({'path':path,'generated_tokens':len(ids),'finish_reason':g['finish_reason'],
                         'configured_model_eos_emissions':ids.count(151645),'tokenizer_eos_emissions':ids.count(151643),
                         'first_tokenizer_eos_generated_index_zero_based':first,
                         'text_through_first_tokenizer_eos':prefix,
                         'text_after_last_think_before_first_tokenizer_eos':prefix.rsplit('</think>',1)[-1] if prefix is not None else None})
result={'model_eos':151645,'tokenizer_eos':151643,'default_policy_unchanged':True,
        'explicit_workaround':'--stop-at-tokenizer-eos adds 151643 while retaining 151645; not an original-runtime fidelity claim',
        'observations':observations}
(out/'eos-observation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps([{k:v for k,v in x.items() if not k.startswith('text_')} for x in observations],indent=2))
