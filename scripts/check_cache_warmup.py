"""Validate ownership, disposal and interruption of the bounded prefix refresh."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from portable.model import load_model
torch.set_num_threads(4);m=load_model('cpu');cache=m.new_cache();checks={}
initial=torch.tensor([[1,2,3]]);m(initial,cache);initial.fill_(99)
checks['stored_ids_are_owned_copy']=cache.warmup_ids.tolist()==[[1,2,3]]
captures={};got=m(torch.tensor([[4,5]]),cache,lambda name,value:captures.update({name:list(value.shape)}))
expected=m.reference_hidden(torch.tensor([[1,2,3,4,5]]))[:,-2:]
checks['refreshed_output_matches_prefix']=torch.equal(got,expected) and cache.length==5
checks['captures_only_new_tokens']=all(shape[1]==2 for shape in captures.values())
def interrupt(name,value):
    if name=='layer_2':raise RuntimeError('Controlled warmup interruption')
try:m(torch.tensor([[6]]),cache,interrupt)
except RuntimeError:pass
checks['refresh_failure_invalidates_original_cache']=not cache.valid
try:m(torch.tensor([[6]]),cache)
except ValueError:checks['invalid_cache_rejected']=True
else:checks['invalid_cache_rejected']=False
cache.reset();checks['reset_discards_buffer_and_states']=cache.warmup_ids is None and all(x is None for x in cache.keys) and cache.length==0 and cache.valid
m(torch.tensor([[1,2,3]]),cache)
ids=torch.arange(4,18)[None];got=m(ids,cache)
checks['crossing_threshold_discards_buffer']=cache.length==17 and cache.warmup_ids is None
checks['crossing_threshold_output_exact']=torch.equal(got,m.reference_hidden(torch.arange(1,18)[None])[:,-14:])
result={'checks':checks,'passed':all(checks.values()),'warmup_limit':m.cache_warmup_tokens,'minimum_query_rows':m.attention_min_query_rows}
(ROOT/'evidence/03b-cache/warmup-lifecycle.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2));raise SystemExit(0 if result['passed'] else 1)
