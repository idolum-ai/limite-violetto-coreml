"""Exact failed-prompt trajectories and controlled input/cache GPU-policy replays.

No model, qualified runtime, validator, or acceptance limit is changed.
"""
from __future__ import annotations
import argparse,gc,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
SOURCE=ROOT/'artifacts/04d-cpu/full-candidate-6'
CACHE=ROOT/'artifacts/05c-compiled/full-cache-1'
SM=ROOT/'evidence/04d-cpu/full-candidate-6/package-identity.json'
CM=ROOT/'evidence/05c-compiled/full-cache-1/compiled-identity.json'
SID='5a3e4c7a571e4fc3532377d9b9bb0da023f499c10709216fee7e1001bf4ad2c6'
CID='a838dd36e8f87c6b9fc0dd7865d0912651c6ab3399fa299d10ff12d788bcb031'

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,value):
    t=path.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');t.replace(path)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--attempt',required=True);p.add_argument('--protocol',required=True);a=p.parse_args()
    if not a.attempt or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in a.attempt):p.error('Path-safe fresh attempt required')
    protocol_path=ROOT/a.protocol;protocol=read(protocol_path)
    assert protocol['script_sha256']==sha(__file__) and protocol['chunks']==[32,12] and protocol['groups']==list(range(5))
    for path,digest in protocol['pinned_files'].items():assert sha(ROOT/path)==digest,path
    out=ROOT/'evidence/06c-gpu-localization'/a.attempt;out.mkdir(parents=True,exist_ok=False)
    (out/'source-snapshot.py').write_bytes(Path(__file__).read_bytes());(out/'protocol.json').write_bytes(protocol_path.read_bytes())
    r={'status':'starting','script_sha256':sha(__file__),'protocol_sha256':sha(protocol_path),'protocol':protocol,'loads':{},'embeddings':[], 'groups':[], 'reference_checks':[], 'files':{},'actual_device_placement':'unverified; compute policy only','thresholds_changed':False}
    def update():save(out/'results.json',r)
    update();models={};owned={}
    try:
        import numpy as np
        import torch
        import coremltools as ct
        torch.set_num_threads(4)
        from coreml_port.compiled_identity import verify_compiled_identity
        from coreml_port.prediction_guard import install,stats
        from scripts.summarize_conditional_milestone import qualified_context
        from scripts.runtime_provenance import runtime_provenance
        from scripts.probe_gpu_group0 import compare_array
        from scripts.probe_full_model import metrics
        from portable.model import load_model
        ctx=qualified_context();assert not ctx['errors'],ctx['errors'];r['qualified_sources']=ctx['expected_sources'];r['runtime_provenance']=runtime_provenance()
        ident=verify_compiled_identity(CACHE,CM,SOURCE,SM);assert ident['aggregate_sha256']==CID and ident['source_candidate_identity']==SID
        r['source_identity']=SID;r['compiled_identity']=CID
        ids=read(ROOT/'evidence/01-reference/local-input-validation.json')['cases'][0]['input_ids'];assert len(ids)==44
        r['input_ids']=ids
        def archive(label,arrays):
            path=out/(label+'.npz');path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**arrays)
            info={'path':str(path.relative_to(ROOT)),'sha256':sha(path),'shapes':{n:list(v.shape) for n,v in arrays.items()},'dtypes':{n:str(v.dtype) for n,v in arrays.items()}}
            r['files'][label]=info;return info
        def equal_maps(x,y):return set(x)==set(y) and all(x[n].shape==y[n].shape and x[n].dtype==y[n].dtype and np.array_equal(x[n],y[n]) for n in x)
        def map_metrics(x,y):
            assert set(x)==set(y)
            return {n:compare_array(x[n],y[n]) for n in sorted(x)}
        # Reproduce the original portable reference call, retaining only needed checkpoints.
        names={'embedding_norm',*[f'layer_{4*g+3}' for g in range(5)]};reference={}
        print('loading portable reference',flush=True);ref=load_model()
        with torch.inference_mode():ref(torch.tensor([ids]),capture=lambda n,v:reference.update({n:v.detach().float().numpy().copy()}) if n in names else None)
        assert set(reference)==names;archive('portable-reference',reference);del ref;gc.collect()
        install();before=stats();assert before['plain_calls']==before['state_calls']==before['gil_releases']==0
        states_names={str(g):ident['packages'][str(g)]['state_names'] for g in range(5)}
        assert all(len(v)==8 and len(set(v))==8 for v in states_names.values())
        norm={}
        for name in ['embedding',*map(str,range(5))]:
            spec=ct.models.utils.load_spec(str(SOURCE/f'{name}.mlpackage'));norm[name]=any(x.name=='norm_divisor' for x in spec.description.input)
        for policy in ['CPU','GPU']:
            models[policy]={};owned[policy]={};r['loads'][policy]={}
            for name in ['embedding',*map(str,range(5))]:
                start=time.perf_counter();model=ct.models.CompiledMLModel(str(CACHE/f'{name}.mlmodelc'),compute_units=ct.ComputeUnit.CPU_ONLY if policy=='CPU' else ct.ComputeUnit.CPU_AND_GPU)
                models[policy][name]=model;r['loads'][policy][name]=time.perf_counter()-start
                if name!='embedding':
                    owned[policy][name]=model.make_state()
                    assert all(np.count_nonzero(owned[policy][name].read_state(n))==0 for n in states_names[name])
                update();print('loaded',policy,name,flush=True)
        captures={policy:{n:[] for n in names} for policy in models};counts={'plain':0,'state':0};disposed=0;restored=0
        def state_arrays(state,group):return {n:state.read_state(n).copy() for n in states_names[group]}
        def predict(policy,name,inputs,state=None):
            counts['plain' if state is None else 'state']+=1
            payload={**inputs,**({'norm_divisor':np.array([1280.],np.float32)} if norm[name] else {})}
            got=models[policy][name].predict(payload,**({'state':state} if state is not None else {}))
            assert all(np.isfinite(x).all() for x in got.values())
            return {n:v.copy() for n,v in got.items()}
        for chunk_index,(start,count) in enumerate([(0,32),(32,12)]):
            paths={};embeddings={}
            for policy in models:
                emb=predict(policy,'embedding',{'ids':np.array([ids[start:start+count]],np.int32)})
                embeddings[policy]=emb;archive(f'chunk{chunk_index}/embedding-{policy}',emb)
                captures[policy]['embedding_norm'].append(emb['x'])
                paths[policy]={'x':emb['x'],'ve':emb['ve'],'positions':np.arange(start,start+count,dtype=np.int32),'tap0':emb['x'],'tap12':emb['x'],'tap23':emb['x']}
            r['embeddings'].append({'chunk':chunk_index,'CPU_vs_GPU':map_metrics(embeddings['CPU'],embeddings['GPU'])});update()
            for group_int in range(5):
                group=str(group_int);key=f'chunk{chunk_index}/group{group}';before_states={};inputs={};outputs={};after_states={}
                row={'chunk':chunk_index,'group':group_int,'start':start,'count':count,'trajectories':{},'replays':[],'contrasts':{}}
                # Own trajectories preserve cache and dataflow across chunks and groups.
                for policy in models:
                    inputs[policy]={n:v.copy() for n,v in paths[policy].items()}
                    if norm[group]:inputs[policy]['norm_divisor']=np.array([1280.],np.float32)
                    before_states[policy]=state_arrays(owned[policy][group],group)
                    outputs[policy]=predict(policy,group,inputs[policy],owned[policy][group]);after_states[policy]=state_arrays(owned[policy][group],group)
                    row['trajectories'][policy]={kind:archive(f'{key}/own-{policy}/{kind}',arr) for kind,arr in [('inputs',inputs[policy]),('state-before',before_states[policy]),('outputs',outputs[policy]),('state-after',after_states[policy])]}
                    captures[policy][f'layer_{group_int*4+3}'].append(outputs[policy]['y'])
                    paths[policy]['x']=outputs[policy]['y'];paths[policy]['tap23']=outputs[policy]['tap23_out']
                    if group_int==2:paths[policy]['tap12']=outputs[policy]['y']
                row['trajectory_output_comparison']=map_metrics(outputs['CPU'],outputs['GPU'])
                row['trajectory_written_cache_comparison']=map_metrics({n:v[:,start:start+count] for n,v in after_states['CPU'].items()},{n:v[:,start:start+count] for n,v in after_states['GPU'].items()})
                if chunk_index==0:assert equal_maps(before_states['CPU'],before_states['GPU'])
                variants={}
                for input_policy in models:
                    for state_policy in (['CPU'] if chunk_index==0 else ['CPU','GPU']):
                        for compute_policy in models:
                            label=f'I{input_policy}-S{state_policy}-P{compute_policy}';state=models[compute_policy][group].make_state()
                            try:
                                assert all(np.count_nonzero(state.read_state(n))==0 for n in states_names[group])
                                for n,v in before_states[state_policy].items():state.write_state(n,v)
                                assert equal_maps(state_arrays(state,group),before_states[state_policy]);restored+=1
                                payload={n:v.copy() for n,v in inputs[input_policy].items()}
                                got=predict(compute_policy,group,payload,state);post=state_arrays(state,group)
                                assert equal_maps(payload,inputs[input_policy]),'Input mutation'
                                variants[(input_policy,state_policy,compute_policy)]=(got,post)
                                expected_own=input_policy==compute_policy and (state_policy==compute_policy or chunk_index==0)
                                repeat_ok=equal_maps(got,outputs[compute_policy]) and equal_maps(post,after_states[compute_policy]) if expected_own else None
                                if expected_own:assert repeat_ok,f'Own-trajectory control failed: {key}/{label}'
                                record={'label':label,'input_trajectory':input_policy,'state_trajectory':state_policy,'compute_policy':compute_policy,'restoration_exact':True,'own_trajectory_control':repeat_ok,
                                        'outputs':archive(f'{key}/replay-{label}/outputs',got),'state-after':archive(f'{key}/replay-{label}/state-after',post)}
                                row['replays'].append(record)
                            finally:del state;disposed+=1
                # Policy contrasts hold both input and cache fixed. Input/cache contrasts hold the remaining factors fixed.
                for ip in models:
                    for sp in (['CPU'] if chunk_index==0 else ['CPU','GPU']):
                        ca,sa=variants[(ip,sp,'CPU')];gb,sb=variants[(ip,sp,'GPU')]
                        row['contrasts'][f'policy-I{ip}-S{sp}']={'outputs':map_metrics(ca,gb),'written_state':map_metrics({n:v[:,start:start+count] for n,v in sa.items()},{n:v[:,start:start+count] for n,v in sb.items()})}
                for cp in models:
                    for sp in (['CPU'] if chunk_index==0 else ['CPU','GPU']):
                        row['contrasts'][f'input-S{sp}-P{cp}']={'outputs':map_metrics(variants[('CPU',sp,cp)][0],variants[('GPU',sp,cp)][0])}
                    if chunk_index==1:
                        for ip in models:row['contrasts'][f'cache-I{ip}-P{cp}']={'outputs':map_metrics(variants[(ip,'CPU',cp)][0],variants[(ip,'GPU',cp)][0])}
                r['groups'].append(row);update();print('captured/replayed',key,flush=True)
                del variants
        prior={c['name']:c for c in read(ROOT/protocol['prior_gpu_result'])['comparisons']}
        for name in sorted(names):
            cpu=np.concatenate(captures['CPU'][name],axis=1);gpu=np.concatenate(captures['GPU'][name],axis=1)
            assert np.array_equal(cpu,reference[name]),f'CPU reference differs: {name}'
            result=metrics(torch.from_numpy(gpu),torch.from_numpy(reference[name]))
            old=prior[f'prompt0/{name}'];agreement={k:abs(result[k]-old[k])<=max(1e-12,abs(old[k])*1e-10) for k in result}
            r['reference_checks'].append({'name':name,'cpu_reference_exact':True,'gpu_metrics':result,'prior_gpu_metrics':{k:old[k] for k in result},'prior_metric_agreement':agreement})
            archive('trajectory-'+name,{'CPU':cpu,'GPU':gpu})
        r['prior_gpu_metrics_reproduced']=all(all(x['prior_metric_agreement'].values()) for x in r['reference_checks'])
        assert r['prior_gpu_metrics_reproduced'],'GPU trajectory did not reproduce prior screen; controls require review'
        r['prediction_guard']=stats();assert counts=={'plain':4,'state':80};assert stats()=={'installed':True,'plain_calls':4,'state_calls':80,'gil_releases':84}
        assert disposed==restored==60;r['fresh_replay_states']={'restored_exact':restored,'python_references_deleted':disposed}
        owned.clear();models.clear();gc.collect();r['trajectory_state_references_cleared']=True
        assert verify_compiled_identity(CACHE,CM,SOURCE,SM)==ident
        assert qualified_context()['expected_sources']==ctx['expected_sources'] and not qualified_context()['errors']
        assert runtime_provenance()==r['runtime_provenance'] and sha(__file__)==r['script_sha256'] and sha(protocol_path)==r['protocol_sha256']
        for path,digest in protocol['pinned_files'].items():assert sha(ROOT/path)==digest,path
        r.update(status='completed',passed=True,diagnostic_completed=True,gpu_numerical_qualification='not_qualified',scope='Exact 44-token failed prompt; two own trajectories through group4 plus shared-input/cache factorial replays. Diagnostic completion, not GPU numerical qualification.')
    except BaseException as e:r.update(status='failed',passed=False,error={'type':type(e).__name__,'message':str(e)});raise
    finally:owned.clear();models.clear();update()
    print(json.dumps({'status':r['status'],'passed':r['passed'],'prior_gpu_metrics_reproduced':r['prior_gpu_metrics_reproduced'],'guard':r['prediction_guard']},indent=2))
if __name__=='__main__':main()
