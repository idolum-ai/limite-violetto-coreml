"""Opt-in generation through a verified persistent compiled CPU cache."""
import argparse
import json
import sys
import torch
from portable.generate import tokenizer
from .generate import generate
from .loader_selection import add_loader_arguments, create_pipeline, loader_provenance

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('prompt');p.add_argument('--model-dir',required=True);p.add_argument('--identity-manifest',required=True)
    add_loader_arguments(p)
    p.add_argument('--max-new-tokens',type=int,default=192);p.add_argument('--sample',action='store_true')
    p.add_argument('--seed',type=int,default=0);p.add_argument('--stop-at-tokenizer-eos',action='store_true')
    p.add_argument('--stream',action='store_true');args=p.parse_args()
    if not args.compiled_dir or not args.compiled_manifest:p.error('Both compiled cache arguments are required.')
    args.compute='CPU_ONLY';args.tiny_prefix_replay=True;torch.set_num_threads(4)
    tok=tokenizer();ids=tok.apply_chat_template([{'role':'user','content':args.prompt}],tokenize=True,add_generation_prompt=True)
    if not isinstance(ids,list):ids=ids['input_ids']
    if ids and isinstance(ids[0],list):ids=ids[0]
    model=create_pipeline(args);stream_ids=[];written=''
    def stream(token):
        nonlocal written
        stream_ids.append(token);text=tok.decode(stream_ids,skip_special_tokens=True)
        if not text.endswith('\ufffd') and text.startswith(written):
            print(text[len(written):],end='',file=sys.stderr,flush=True);written=text
    try:
        result=generate(model,ids,args.max_new_tokens,not args.sample,args.seed,(tok.eos_token_id,) if args.stop_at_tokenizer_eos else (),stream if args.stream else None)
    except KeyboardInterrupt:
        print('\nCancelled; request state disposed.',file=sys.stderr);return 130
    if args.stream:print(file=sys.stderr)
    result.update(text=tok.decode(result['generated_ids'],skip_special_tokens=True),
                  fidelity='This prompt is not independently audited; consult the loader qualification evidence.',
                  loader_provenance=loader_provenance(args),load_seconds=model.load_seconds,
                  model_dir=str(model.model_dir),package_identity_sha256=model.package_identity['aggregate_sha256'])
    print(json.dumps(result,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
