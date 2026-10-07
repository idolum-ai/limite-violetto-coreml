"""Bounded PyTorch generation for Step 3; this is not the Core ML runner."""
import argparse
import json
import time
import torch
from transformers import AutoTokenizer
from .model import ROOT, load_model


def tokenizer():
    return AutoTokenizer.from_pretrained(ROOT/'sources/model',local_files_only=True,trust_remote_code=False)


def choose_token(scores,valid_vocab,greedy,generator,temperature=.6,top_p=.95):
    # Padded embedding rows are not tokenizer IDs; exclude them before sampling.
    scores=scores.flatten()[:valid_vocab].float()
    if greedy:return int(scores.argmax().item())
    if temperature<=0 or not 0<top_p<=1:raise ValueError('Invalid temperature/top_p.')
    probs=(scores.cpu()/temperature).softmax(-1)
    sorted_probs,indices=probs.sort(descending=True)
    remove=sorted_probs.cumsum(-1)-sorted_probs>=top_p
    sorted_probs[remove]=0
    return int(indices[torch.multinomial(sorted_probs,1,generator=generator)].item())


def stop_reason(token,eos,count,maximum):
    if token==eos:return 'eos'
    if count>=maximum:return 'length'
    return None


@torch.inference_mode()
def generate(model,prompt_ids,max_new_tokens=192,greedy=True,seed=0,on_token=None,additional_stop_ids=()):
    if max_new_tokens<1:raise ValueError('max_new_tokens must be positive.')
    if len(prompt_ids)+max_new_tokens>model.capacity:
        raise ValueError('Prompt plus requested generation exceeds context capacity; no truncation.')
    extra_stops=set(additional_stop_ids)
    if any(not isinstance(token,int) or token<0 or token>=model.config.tokenizer_vocab_size for token in extra_stops):
        raise ValueError('Additional stops must be valid integer tokenizer IDs.')
    ids=torch.tensor([prompt_ids],device=model.device,dtype=torch.long)
    model.validate_ids(ids)
    cache=model.new_cache()
    generator=torch.Generator(device='cpu').manual_seed(seed)
    generated=[];timings=[];start=time.perf_counter()
    try:
        for n in range(max_new_tokens):
            t=time.perf_counter()
            hidden=model(ids,cache)
            scores=model.scores(hidden[:,-1:])
            token=choose_token(scores,model.config.tokenizer_vocab_size,greedy,generator)
            timings.append(time.perf_counter()-t)
            generated.append(token)
            if on_token:on_token(token)
            finish=stop_reason(token,model.config.eos_token_id,n+1,max_new_tokens)
            if token in extra_stops and token!=model.config.eos_token_id:finish='stop_token'
            if finish:break
            ids=torch.tensor([[token]],device=model.device,dtype=torch.long)
        return {'prompt_ids':prompt_ids,'generated_ids':generated,'finish_reason':finish,
                'complete':finish in {'eos','stop_token'},'eos_token_id':model.config.eos_token_id,
                'stop_token_ids':sorted(extra_stops|{model.config.eos_token_id}),
                'completion_policy':'published_eos_plus_explicit_stops' if extra_stops else 'published_eos_only',
                'greedy':greedy,'seed':seed,'temperature':None if greedy else .6,
                'top_p':None if greedy else .95,'elapsed_seconds':time.perf_counter()-start,
                'step_seconds':timings,'prompt_tokens':len(prompt_ids),'generated_tokens':len(generated)}
    finally:
        cache.reset()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('prompt');p.add_argument('--device',choices=['cpu','mps'],default='cpu')
    p.add_argument('--max-new-tokens',type=int,default=192)
    p.add_argument('--sample',action='store_true');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--stop-at-tokenizer-eos',action='store_true',help='Explicitly add tokenizer EOS151643; published model EOS151645 remains enabled.')
    args=p.parse_args();torch.set_num_threads(4)
    tok=tokenizer();model=load_model(args.device)
    ids=tok.apply_chat_template([{'role':'user','content':args.prompt}],tokenize=True,add_generation_prompt=True)
    # Transformers 5 may return a BatchEncoding rather than a bare list.
    if not isinstance(ids,list):ids=ids['input_ids']
    if ids and isinstance(ids[0],list):ids=ids[0]
    extra=(tok.eos_token_id,) if args.stop_at_tokenizer_eos else ()
    result=generate(model,ids,args.max_new_tokens,not args.sample,args.seed,additional_stop_ids=extra)
    result['text']=tok.decode(result['generated_ids'],skip_special_tokens=True)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
