"""Reproducible real-checkpoint Step 3 validation; no CUDA fidelity claim."""
import argparse,hashlib,json,resource,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from portable.model import load_model,Cache,upstream,attention_mask
from portable.generate import tokenizer,generate,stop_reason,choose_token
from probe_full_model import metrics


def reference_capture(model,ids,positions):
    captures={};handles=[]
    def keep(name,x):captures[name]=x[:,positions].float().cpu().clone()
    for i,layer in enumerate(model.backbone.layers):
        def hook(module,args,out,i=i):
            keep(f'layer_{i}',out)
            if i in model.config.mudd_layers:
                keep(f'layer_{i}_mixer_attention',args[1]);keep(f'layer_{i}_mixer_residual',args[2])
        handles.append(layer.register_forward_hook(hook))
    handles.append(model.backbone.embed_tokens.register_forward_hook(lambda module,args,out:keep('embedding_norm',upstream.rms_norm(out))))
    try:
        hidden=model.reference_hidden(ids);keep('final_hidden',hidden)
        return hidden,captures
    finally:
        for handle in handles:handle.remove()


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','mps'],default='cpu');p.add_argument('--attempt',required=True)
    p.add_argument('--max-new-tokens',type=int,default=512)
    p.add_argument('--phase',default='03-portable',choices=['03-portable','03b-cache'])
    args=p.parse_args();torch.set_num_threads(4)
    out=ROOT/'evidence'/args.phase/args.attempt;art=ROOT/'artifacts'/args.phase/args.attempt
    out.mkdir();art.mkdir()
    acpath=ROOT/'evidence/03-portable/ACCEPTANCE.json'
    calibration=json.loads((ROOT/f'evidence/03-portable/calibration-{args.device}.json').read_text())
    limits=calibration['frozen_limits']
    result={'device':args.device,'acceptance_sha256':hashlib.sha256(acpath.read_bytes()).hexdigest(),'limits':limits,'checks':[],'comparisons':[],'generation':[], 'independent_cuda_fidelity':'UNVERIFIED'}
    result['source_sha256']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['portable/model.py','portable/generate.py','scripts/validate_portable.py','scripts/probe_full_model.py']}
    def save():
        result['peak_process_rss_gib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30
        (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    def check(name,passed,details=None):
        result['checks'].append({'name':name,'passed':bool(passed),'details':details});save()
        print(name,'PASS' if passed else 'FAIL',flush=True)
    def compare(name,a,b,kind='hidden',exact=False):
        v=metrics(a,b);entry={'name':name,'kind':kind,**v}
        if kind=='scores':
            av=a.flatten()[:151667].float().cpu();bv=b.flatten()[:151667].float().cpu()
            top=av.topk(5);bt=bv.topk(5)
            margin=float(top.values[0]-top.values[1]);match=int(top.indices[0])==int(bt.indices[0])
            entry.update(top1_reference=int(top.indices[0]),top1_candidate=int(bt.indices[0]),top1_match=match,reference_top1_margin=margin,top5_overlap=len(set(top.indices.tolist())&set(bt.indices.tolist())))
            passed=v['max_abs']<=(1e-5 if exact else limits['score_max_abs']) and (match or margin<=2*limits['score_max_abs'])
        else:
            passed=v['max_abs']<=1e-5 if exact else v['relative_rmse']<=limits['hidden_relative_rmse'] and v['cosine']>=limits['hidden_min_cosine']
        entry['passed']=passed;result['comparisons'].append(entry)
        return passed
    def archive(name,a,b):
        arrays={f'reference_{k}':v.numpy() for k,v in a.items()}
        arrays.update({f'candidate_{k}':v.numpy() for k,v in b.items()})
        np.savez_compressed(art/f'{name}.npz',**arrays)
    print('Loading model',flush=True);model=load_model(args.device)
    (out/'weight-mapping.json').write_text(json.dumps(model.load_report,indent=2)+'\n')
    check('exhaustive_weight_mapping',model.load_report['tensor_count']==743 and model.load_report['parameter_count']==1035254176)
    check('reference_calibration_valid',limits['calibration_valid'])
    tok=tokenizer()
    cases=json.loads((ROOT/'evidence/01-reference/local-input-validation.json').read_text())['cases']
    for i,case in enumerate(cases):
        ids=tok.apply_chat_template(case['messages'],tokenize=True,add_generation_prompt=True)
        if not isinstance(ids,list):ids=ids['input_ids']
        if ids and isinstance(ids[0],list):ids=ids[0]
        check(f'prompt_{i}_exact_ids',ids==case['input_ids'])
    check('published_eos_selected',model.config.eos_token_id==151645 and tok.eos_token_id==151643)
    sequences=[(f'prompt_{i}',case['input_ids'],None) for i,case in enumerate(cases)]
    sequences.append(('window',(cases[0]['input_ids']*24)[:1026],[1023,1024,1025]))
    for name,token_ids,selected in sequences:
        print('Sequence',name,len(token_ids),flush=True);t=time.perf_counter()
        ids=torch.tensor([token_ids],device=model.device);n=len(token_ids)
        selected=list(range(n)) if selected is None else selected
        ref,rc=reference_capture(model,ids,selected)
        pc={}
        portable=model(ids,capture=lambda key,value:pc.update({key:value[:,selected].float().cpu().clone()}))
        full_checks=[compare(name+'/full/'+key,value,pc[key],exact=True) for key,value in rc.items()]
        refscore=model.scores(ref[:,-1:]);pscore=model.scores(portable[:,-1:])
        full_checks.append(compare(name+'/full/scores',refscore,pscore,'scores',True))
        check(name+'_full_prefix_48_layers',all(full_checks))
        archive(name+'-full',rc,pc)
        np.savez_compressed(art/f'{name}-scores.npz',reference=refscore.float().cpu().numpy(),portable=pscore.float().cpu().numpy())
        cache=model.new_cache();cc={};chunks=[]
        sizes=([256,256,256,256,1,1] if name=='window' else [n-3,1,1,1])
        start=0
        for size in sizes:
            end=start+size;local=[pos-start for pos in selected if start<=pos<end]
            def capture(key,value):
                if local:cc.setdefault(key,[]).append(value[:,local].float().cpu().clone())
            chunk=model(ids[:,start:end],cache,capture);chunks.append(chunk)
            if name=='window' and end in (1024,1025,1026):
                rs=model.scores(ref[:,end-1:end]);cs=model.scores(chunk[:,-1:])
                compare(f'{name}/cache/position_{end-1}_scores',rs,cs,'scores')
                np.savez_compressed(art/f'window-position-{end-1}-scores.npz',reference=rs.float().cpu().numpy(),cached=cs.float().cpu().numpy())
            start=end
        cc={k:torch.cat(v,dim=1) for k,v in cc.items()}
        cache_checks=[compare(name+'/cache/'+key,value,cc[key]) for key,value in rc.items()]
        cache_checks.append(compare(name+'/cache/final_scores',refscore,model.scores(chunks[-1][:,-1:]),'scores'))
        check(name+'_cached_layers',all(cache_checks) and cache.length==n)
        archive(name+'-cached',rc,cc)
        # Read only the populated cache, never uninitialized future storage.
        check(name+'_cache_shape_dtype',all(k.shape==(1,2048,2,128) and k.dtype==torch.bfloat16 for k in cache.keys))
        if name=='prompt_0':
            saved=portable.float().cpu();cache.reset()
            reset=model(ids,cache)
            check('reset_matches_fresh',torch.equal(reset.float().cpu(),saved) and cache.length==n)
            other=torch.tensor([cases[1]['input_ids']],device=model.device)
            cache.reset();reuse=model(other,cache);fresh=model(other,model.new_cache())
            check('different_request_matches_fresh',torch.equal(reuse,fresh))
        cache.reset()
        result.setdefault('sequence_seconds',{})[name]=time.perf_counter()-t;save()
    # Teacher forcing holds the text fixed so generation drift cannot hide score drift.
    print('Teacher-forced continuation',flush=True)
    prefix=cases[0]['input_ids'];continuation=tok.encode('17 * 23 = 391. Therefore, the answer is \\boxed{391}.',add_special_tokens=False)
    cache=model.new_cache();model(torch.tensor([prefix],device=model.device),cache)
    teacher=[]
    for i,token in enumerate(continuation[:12]):
        prefix=prefix+[token];cur=model(torch.tensor([[token]],device=model.device),cache)
        ref=model.reference_hidden(torch.tensor([prefix],device=model.device))
        teacher.append(compare(f'teacher_{i}/hidden',ref[:,-1:],cur))
        teacher.append(compare(f'teacher_{i}/scores',model.scores(ref[:,-1:]),model.scores(cur),'scores'))
    check('teacher_forced_continuation',all(teacher));cache.reset()
    # Behavioral rejection must leave the cache untouched.
    def rejects(name,fn):
        try:fn()
        except ValueError:check(name,True)
        else:check(name,False)
    ids=torch.tensor([[1,2,3]],device=model.device)
    for name,bad in [('negative',[-1]),('padded',[151667]),('out_of_range',[151680])]:
        rejects('reject_'+name,lambda bad=bad:model(torch.tensor([bad],device=model.device),cache))
    rejects('reject_overflow',lambda:model(torch.ones((1,2049),device=model.device,dtype=torch.long),cache))
    rejects('reject_wrong_owner',lambda:model(ids,Cache(object(),2048,48)))
    check('rejections_did_not_mutate_cache',cache.length==0 and cache.valid and all(k is None for k in cache.keys))
    def interrupt(key,value):
        if key=='layer_2':raise RuntimeError('Injected interrupted forward for reset validation')
    try:model(ids,cache,interrupt)
    except RuntimeError:pass
    check('interrupted_cache_invalidated',not cache.valid)
    rejects('reject_interrupted_cache',lambda:model(ids,cache));cache.reset()
    check('reset_after_interruption',torch.equal(model(ids,cache),model(ids)))
    rejects('reject_generation_overflow',lambda:generate(model,[1]*2040,20))
    check('stop_semantics',stop_reason(151645,151645,1,10)=='eos' and stop_reason(151643,151645,1,10) is None and stop_reason(1,151645,10,10)=='length')
    synthetic=torch.zeros(151680);synthetic[151679]=1e6;synthetic[123]=100
    check('padded_generation_rows_excluded',choose_token(synthetic,151667,True,None)==123)
    # Check the exact inclusive mask condition used by the implementation.
    local=attention_mask(torch.tensor([1024,1025]),1026,1024)
    glob=attention_mask(torch.tensor([1024,1025]),1026,-1)
    check('window_boundary_inclusive',local[0,0] and not local[1,0] and not local[0,1025] and local.sum(1).tolist()==[1025,1025] and glob[:,0].all() and glob.sum(1).tolist()==[1025,1026])
    cache.reset()
    limited=generate(model,cases[0]['input_ids'],1,True)
    check('actual_generation_length_stop',limited['finish_reason']=='length' and not limited['complete'] and limited['generated_tokens']==1)
    print('Real greedy generation',flush=True)
    for i,case in enumerate(cases):
        sample=generate(model,case['input_ids'],args.max_new_tokens,True)
        sample['prompt']=case['messages'][0]['content'];sample['text']=tok.decode(sample['generated_ids'],skip_special_tokens=True)
        result['generation'].append(sample);save()
        print('GEN',i,sample['finish_reason'],sample['generated_tokens'],sample['text'],flush=True)
    result['all_local_checks_passed']=all(c['passed'] for c in result['checks']) and all(c['passed'] for c in result['comparisons'])
    save();print('All local checks passed:',result['all_local_checks_passed'],flush=True)


if __name__=='__main__':main()
