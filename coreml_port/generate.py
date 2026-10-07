"""Experimental full Core ML generation, with explicit stop and stream policy."""
import argparse,json,sys,time
import torch
from portable.generate import tokenizer,choose_token
from .tiny_prefix_runtime import select_pipeline

def generate(model,ids,maximum=192,greedy=True,seed=0,extra_stops=(),on_token=None):
    if type(maximum) is not int or maximum<1:raise ValueError('Maximum must be a positive integer.')
    if len(ids)+maximum>model.capacity:raise ValueError('Prompt plus generation exceeds context; no truncation.')
    if any(type(i) is not int or i<0 or i>=model.valid_vocab for i in extra_stops):raise ValueError('Invalid stop ID.')
    request=model.request();generated=[];times=[];begin=time.perf_counter();g=torch.Generator().manual_seed(seed);stops={model.eos,*extra_stops}
    try:
        for i in range(maximum):
            t=time.perf_counter();scores=model.forward(ids if i==0 else [generated[-1]],request)
            token=choose_token(torch.from_numpy(scores),model.valid_vocab,greedy,g);generated.append(token);times.append(time.perf_counter()-t)
            if on_token:on_token(token)
            if token in stops:break
        finish=('eos' if token==model.eos else 'stop_token') if token in stops else 'length'
        return {'backend':'Core ML','compute_units':model.compute,'runtime_adapter':getattr(model,'runtime_adapter','standard_pipeline'),'fidelity':'UNQUALIFIED','prompt_ids':ids,'generated_ids':generated,'generated_tokens':len(generated),'finish_reason':finish,'complete':finish!='length','stop_token_ids':sorted(stops),'completion_policy':'published_eos_plus_explicit_stops' if extra_stops else 'published_eos_only','greedy':greedy,'seed':seed,'temperature':None if greedy else .6,'top_p':None if greedy else .95,'step_seconds':times,'elapsed_seconds':time.perf_counter()-begin}
    finally:request.close()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('prompt');p.add_argument('--max-new-tokens',type=int,default=192);p.add_argument('--compute',choices=['CPU_ONLY','CPU_AND_GPU'],default='CPU_ONLY');p.add_argument('--model-dir',help='Explicit candidate package directory; requires --identity-manifest.');p.add_argument('--identity-manifest',help='Verify all candidate packages before loading.');p.add_argument('--tiny-prefix-replay',action='store_true',help='Opt into CPU-equivalent cached-prefix replay below 16 tokens.');p.add_argument('--sample',action='store_true');p.add_argument('--seed',type=int,default=0);p.add_argument('--stop-at-tokenizer-eos',action='store_true');p.add_argument('--stream',action='store_true',help='Stream text to stderr; JSON result remains on stdout.');args=p.parse_args();torch.set_num_threads(4)
    if (args.model_dir is None)!=(args.identity_manifest is None):p.error('--model-dir and --identity-manifest must be supplied together')
    if args.tiny_prefix_replay and args.compute!='CPU_ONLY':p.error('--tiny-prefix-replay requires --compute CPU_ONLY')
    tok=tokenizer();ids=tok.apply_chat_template([{'role':'user','content':args.prompt}],tokenize=True,add_generation_prompt=True)
    if not isinstance(ids,list):ids=ids['input_ids']
    if ids and isinstance(ids[0],list):ids=ids[0]
    model=select_pipeline(args.tiny_prefix_replay)(args.compute,args.model_dir,args.identity_manifest);stream_ids=[];written=''
    def stream(token):
        nonlocal written
        stream_ids.append(token);text=tok.decode(stream_ids,skip_special_tokens=True)
        if not text.endswith('\ufffd') and text.startswith(written):print(text[len(written):],end='',file=sys.stderr,flush=True);written=text
    try:result=generate(model,ids,args.max_new_tokens,not args.sample,args.seed,(tok.eos_token_id,) if args.stop_at_tokenizer_eos else (),stream if args.stream else None)
    except KeyboardInterrupt:print('\nCancelled; request state disposed.',file=sys.stderr);return 130
    if args.stream:print(file=sys.stderr)
    result['text']=tok.decode(result['generated_ids'],skip_special_tokens=True);result['load_seconds']=model.load_seconds;result['model_dir']=str(model.model_dir);result['package_identity_sha256']=model.package_identity['aggregate_sha256'] if model.package_identity else None;print(json.dumps(result,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
