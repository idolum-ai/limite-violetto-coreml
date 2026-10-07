"""Matched long-prefix/window/capacity gate for an explicitly bound candidate."""
import argparse,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from coreml_port.loader_selection import add_loader_arguments, create_pipeline, runtime_adapter_name, loader_provenance
from portable.model import load_model
from scripts.probe_full_model import metrics
from scripts.runtime_provenance import runtime_provenance
p=argparse.ArgumentParser();p.add_argument('--model-dir',required=True);p.add_argument('--identity-manifest',required=True);p.add_argument('--evidence-dir',required=True);p.add_argument('--fixture',choices=['canonical-cycle','seeded'],default='canonical-cycle');p.add_argument('--tiny-prefix-replay',action='store_true',help='Opt into CPU-equivalent cached-prefix replay below 16 tokens.');add_loader_arguments(p);a=p.parse_args()
runtime_adapter=runtime_adapter_name(a)
torch.set_num_threads(4);out=ROOT/a.evidence_dir;out.mkdir(parents=True,exist_ok=False)
acceptance=hashlib.sha256((ROOT/'evidence/03-portable/ACCEPTANCE.json').read_bytes()).hexdigest()
if acceptance!='cb4d315266126df4a98194f73f8c10671310717c06ebfd191d3dac2334bf4233':raise RuntimeError('Frozen acceptance changed.')
r={'acceptance_sha256':acceptance,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_sha256':{'coreml_port/tiny_prefix_runtime.py':hashlib.sha256((ROOT/'coreml_port/tiny_prefix_runtime.py').read_bytes()).hexdigest(),'coreml_port/runtime.py':hashlib.sha256((ROOT/'coreml_port/runtime.py').read_bytes()).hexdigest()},'runtime_provenance':runtime_provenance(),'loader_provenance':loader_provenance(a),'compute':'CPU_ONLY','runtime_adapter':runtime_adapter,'thresholds_changed':False,'comparisons':[],'checks':{},'status':'starting'}
def save():(out/'results.json').write_text(json.dumps(r,indent=2)+'\n')
save();start=time.time();cm=create_pipeline(a);r['package_identity']=cm.package_identity;r['model_dir']=str(cm.model_dir);save();ref=load_model()
case=json.loads((ROOT/'evidence/01-reference/local-input-validation.json').read_text())['cases'][0]['input_ids']
prefix=(case*24)[:1026] if a.fixture=='canonical-cycle' else torch.randint(100,10000,(1026,),generator=torch.Generator().manual_seed(20260922)).tolist()
ids=prefix+[100]*1022;r['fixture']=a.fixture;r['input_ids_sha256']=hashlib.sha256(json.dumps(ids).encode()).hexdigest()
(out/'input-ids.json').write_text(json.dumps(ids)+'\n')
captured={}
def tap(name,value):
    if name=='final_hidden':captured[name]=value.copy()
def compare(label,got,want,hidden,refhidden):
    s=metrics(torch.from_numpy(got),want);h=metrics(torch.from_numpy(hidden),refhidden)
    actual=int(np.argmax(got.flatten()[:151667]));top=want.flatten()[:151667].topk(5);margin=float(top.values[0]-top.values[1]);ranking=actual==int(top.indices[0]) or margin<=.08
    overlap=len(set(np.argsort(got.flatten()[:151667])[-5:].tolist())&set(top.indices.tolist()))
    record={'name':label,'scores':s,'hidden':h,'top1_candidate':actual,'top1_reference':int(top.indices[0]),'top5_overlap':overlap,'reference_margin':margin,'ranking_pass':ranking,'pass':s['max_abs']<=.04 and h['relative_rmse']<=.01 and h['cosine']>=.995 and ranking}
    r['comparisons'].append(record);save();print(label,'score max_abs',s['max_abs'],'pass',record['pass'],flush=True)
with torch.inference_mode():
    req=cm.request();last=0
    for end in [1023,1024,1025,1026,2048]:
        got=cm.forward(ids[last:end],req,tap);h=ref.reference_hidden(torch.tensor([ids[:end]]))[:,-1:];want=ref.scores(h)
        compare(f'prefix{end}',got,want,captured['final_hidden'][:,-1:],h)
        if end==1026:boundary=got.copy()
        last=end
    before=req.length
    try:cm.forward([100],req);r['checks']['overflow_rejected_before_mutation']=False
    except ValueError:r['checks']['overflow_rejected_before_mutation']=req.length==before and req.valid
    req.close();r['checks']['capacity_request_disposed']=req.closed and not req.states;save()
    fresh=cm.request();got=cm.forward(ids[:1026],fresh,tap);h=ref.reference_hidden(torch.tensor([ids[:1026]]))[:,-1:];compare('fresh1026',got,ref.scores(h),captured['final_hidden'][:,-1:],h)
    r['checks']['different_partition_boundary_repeat_exact']=bool(np.array_equal(boundary,got));fresh.close();r['checks']['fresh_request_disposed']=fresh.closed and not fresh.states
r.update(status='completed',elapsed_seconds=time.time()-start,numerical_pass=all(x['pass'] for x in r['comparisons']),checks_pass=all(r['checks'].values()));save();print(r['numerical_pass'],r['checks_pass'],flush=True)
