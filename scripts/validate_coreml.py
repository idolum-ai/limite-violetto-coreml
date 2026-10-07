"""Real complete-model comparison and request-lifecycle gates; fixed tolerances."""
import argparse,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from portable.model import load_model
from coreml_port.runtime import Request
from coreml_port.loader_selection import add_loader_arguments, create_pipeline, runtime_adapter_name, loader_provenance
from coreml_port.generate import generate
from scripts.probe_full_model import metrics
from scripts.runtime_provenance import runtime_provenance
p=argparse.ArgumentParser();p.add_argument('--compute',default='CPU_ONLY');p.add_argument('--evidence-dir',default='evidence/04-coreml');p.add_argument('--model-dir');p.add_argument('--identity-manifest');p.add_argument('--tiny-prefix-replay',action='store_true',help='Opt into CPU-equivalent cached-prefix replay below 16 tokens.');add_loader_arguments(p);args=p.parse_args();torch.set_num_threads(4)
if args.tiny_prefix_replay and args.compute!='CPU_ONLY':p.error('--tiny-prefix-replay requires --compute CPU_ONLY')
runtime_adapter=runtime_adapter_name(args)
path=ROOT/args.evidence_dir/f'validation-{args.compute}.json';path.parent.mkdir(exist_ok=True,parents=True)
attempt={'status':'starting','compute':args.compute,'runtime_adapter':runtime_adapter,'model_dir':args.model_dir,'identity_manifest':args.identity_manifest,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'runtime_provenance':runtime_provenance(),'loader_provenance':loader_provenance(args)}
attempt['source_hashes']={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['coreml_port/runtime.py','coreml_port/tiny_prefix_runtime.py','coreml_port/package_identity.py','portable/model.py','evidence/03-portable/ACCEPTANCE.json','evidence/01-reference/local-input-validation.json']}
(path.parent/f'validation-{args.compute}-attempt.json').write_text(json.dumps(attempt,indent=2)+'\n')
if attempt['source_hashes']['evidence/03-portable/ACCEPTANCE.json']!='cb4d315266126df4a98194f73f8c10671310717c06ebfd191d3dac2334bf4233':
    raise RuntimeError('Frozen acceptance file changed; refusing this gate run.')
start=time.time();cm=create_pipeline(args);print('loaded',sum(cm.load_seconds.values()),flush=True);cases=json.loads((ROOT/'evidence/01-reference/local-input-validation.json').read_text())['cases']
r={'compute_units':args.compute,'runtime_adapter':runtime_adapter,'runtime_provenance':runtime_provenance(),'loader_provenance':loader_provenance(args),'actual_device_placement':'not profiled','load_seconds':cm.load_seconds,'comparisons':[],'checks':{},'thresholds_changed':False}
r['model_dir']=str(cm.model_dir);r['package_identity']=cm.package_identity
def save():path.write_text(json.dumps(r,indent=2)+'\n')
save();ref=load_model()
def compare(name,got,want,score=False):
    v=metrics(torch.from_numpy(got),want);v.update(name=name)
    if score:
        v['pass']=v['max_abs']<=.04;v['top1_candidate']=int(np.argmax(got.flatten()[:151667]));v['top1_reference']=int(want.flatten()[:151667].argmax());v['top1_match']=v['top1_candidate']==v['top1_reference'];top=want.flatten()[:151667].topk(5);v['reference_margin']=float(top.values[0]-top.values[1]);v['top5_overlap']=len(set(np.argsort(got.flatten()[:151667])[-5:].tolist()) & set(top.indices.tolist()));v['ranking_pass']=v['top1_match'] or v['reference_margin']<=.08;v['pass']=v['pass'] and v['ranking_pass']
    else:v['pass']=v['relative_rmse']<=.01 and v['cosine']>=.995
    r['comparisons'].append(v)
with torch.inference_mode():
    for index,case in enumerate(cases):
        ids=case['input_ids'];captured={};reference={};req=cm.request()
        def capture(n,v):captured.setdefault(n,[]).append(v.copy())
        got=cm.forward(ids,req,capture);hidden=ref(torch.tensor([ids]),capture=lambda n,v:reference.update({n:v.clone()}));want=ref.scores(hidden[:,-1:]);compare(f'prompt{index}/scores',got,want,True)
        for n,parts in captured.items():compare(f'prompt{index}/{n}',np.concatenate(parts,axis=1),reference[n])
        first=got.copy()
        for step in range(4):
            token=int(want.flatten()[:151667].argmax());ids=ids+[token];got=cm.forward([token],req);want=ref.scores(ref.reference_hidden(torch.tensor([ids]))[:,-1:]);compare(f'prompt{index}/teacher{step}',got,want,True)
        req.close();r['checks'][f'request{index}_disposed']=req.closed and not req.states
        fresh=cm.request();again=cm.forward(case['input_ids'],fresh);r['checks'][f'request{index}_repeat_exact']=bool(np.array_equal(first,again));fresh.close();save();print('case',index,'done',flush=True)
    ids=cases[0]['input_ids'];req=cm.request()
    def reject(key,fn):
        try:fn();r['checks'][key]=False
        except (ValueError,RuntimeError):r['checks'][key]=True
    reject('padded_id_rejected',lambda:cm.forward([151667],req));reject('empty_rejected',lambda:cm.forward([],req));reject('foreign_owner_rejected',lambda:cm.forward([1],Request(object())))
    req.length=2048;reject('overflow_rejected',lambda:cm.forward([1],req));req.length=0
    def cancel(n,v):raise KeyboardInterrupt()
    try:cm.forward(ids,req,cancel)
    except KeyboardInterrupt:r['checks']['interrupt_invalidates']=not req.valid
    reject('invalid_request_rejected',lambda:cm.forward([1],req));req.close();reject('closed_request_rejected',lambda:cm.forward([1],req))
    # Actual Core ML prediction determines the explicit stop ID, no mock scores.
    first_req=cm.request();first_token=int(cm.forward(ids,first_req).flatten()[:151667].argmax());first_req.close();seen=[]
    result=generate(cm,ids,3,extra_stops=(first_token,),on_token=seen.append);r['checks']['explicit_stop']=result['complete'] and len(result['generated_ids'])==1;r['checks']['stream_callback']=seen==result['generated_ids'];r['stop_test']=result
    saved_request=cm.request
    owned=[]
    def tracked():q=saved_request();owned.append(q);return q
    cm.request=tracked
    def interrupt(token):raise KeyboardInterrupt()
    try:generate(cm,ids,3,on_token=interrupt)
    except KeyboardInterrupt:r['checks']['generation_cancellation_disposes']=owned[-1].closed and not owned[-1].states
    reject('generation_budget_rejected',lambda:generate(cm,ids,2048))
    cm.request=saved_request
from coreml_port.cleanup_guard import stats
r['cleanup_guard']=stats()
r['elapsed_seconds']=time.time()-start;r['numerical_pass']=all(c['pass'] for c in r['comparisons']);r['lifecycle_pass']=all(r['checks'].values());save();print(json.dumps({'numerical_pass':r['numerical_pass'],'lifecycle_pass':r['lifecycle_pass'],'comparisons':len(r['comparisons']),'checks':len(r['checks'])}),flush=True)
attempt.update(status='completed',numerical_pass=r['numerical_pass'],lifecycle_pass=r['lifecycle_pass']);(path.parent/f'validation-{args.compute}-attempt.json').write_text(json.dumps(attempt,indent=2)+'\n')
