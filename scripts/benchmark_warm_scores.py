"""Warm-score variant exact replay followed by balanced ABBA group-call timings."""
from __future__ import annotations
import argparse, hashlib, json, os, platform, statistics, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def inventory(root):return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
def validate_inputs(inputs):
    import numpy as np
    shapes={'x':(1,1,1280),'ve':(1,1,2,128),'positions':(1,), 'tap0':(1,1,1280),'tap12':(1,1,1280),'tap23':(1,1,1280),'norm_divisor':(1,)}
    if set(inputs)!=set(shapes):raise ValueError('Unexpected input names')
    for n,shape in shapes.items():
        value=inputs[n]
        if value.shape!=shape or value.dtype!=(np.int32 if n=='positions' else np.float32):raise ValueError(f'Invalid input shape/dtype: {n}')
    if not 32<=int(inputs['positions'][0])<2048:raise ValueError('Requires one warm query at position 32..2047')
    if not np.array_equal(inputs['norm_divisor'],np.array([1280.],np.float32)):raise ValueError('Unexpected normalization divisor')
def host():
    return {'load_average':list(os.getloadavg()),'power':subprocess.run(['pmset','-g','batt'],capture_output=True,text=True).stdout,'thermal':subprocess.run(['pmset','-g','therm'],capture_output=True,text=True).stdout,'platform':platform.platform()}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--attempt',required=True);p.add_argument('--protocol',required=True);a=p.parse_args()
    if not a.attempt or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in a.attempt):p.error('Invalid attempt')
    out=ROOT/'evidence/05f-scores'/a.attempt;out.mkdir(parents=True,exist_ok=False)
    (out/'source-snapshot.py').write_bytes(Path(__file__).read_bytes())
    protocol_path=ROOT/a.protocol;protocol=json.loads(protocol_path.read_text())
    r={'status':'starting','script_sha256':sha(__file__),'protocol_sha256':sha(protocol_path),'protocol':protocol,'correctness':[],'timings':[],'warmup':[],'timing_boundary':'Synchronous CompiledMLModel.predict including bridge/output conversion; fresh state creation/restoration and exact checks excluded'}
    def save():
        tmp=out/'results.tmp';tmp.write_text(json.dumps(r,indent=2)+'\n');tmp.replace(out/'results.json')
    save()
    try:
        import numpy as np
        import coremltools as ct
        from coreml_port.prediction_guard import install,stats
        from scripts.runtime_provenance import runtime_provenance
        from coreml_port.compiled_identity import verify_compiled_identity
        assert protocol['script_sha256']==r['script_sha256'] and protocol['go_median_reduction']==0.10 and protocol['cycles_per_block']==4 and protocol['blocks']==3
        milestone=json.loads((ROOT/'evidence/05c-compiled/milestone-rerun1.json').read_text())
        assert milestone['milestone_pass'] and runtime_provenance()==milestone['current_runtime_provenance']
        r['runtime_provenance']=runtime_provenance()
        buildpath=ROOT/protocol['build'];assert sha(buildpath)==protocol['build_sha256'];build=json.loads(buildpath.read_text())
        assert build['passed'] and build['graph_check']['passed'] and inventory(ROOT/build['compiled'])==build['compiled_files'] and inventory(ROOT/build['package'])==build['variant_files']
        baseline_root=ROOT/'artifacts/05c-compiled/full-cache-1'
        identity=verify_compiled_identity(baseline_root,ROOT/'evidence/05c-compiled/full-cache-1/compiled-identity.json',ROOT/'artifacts/04d-cpu/full-candidate-6',ROOT/'evidence/04d-cpu/full-candidate-6/package-identity.json')
        assert identity['aggregate_sha256']=='a838dd36e8f87c6b9fc0dd7865d0912651c6ab3399fa299d10ff12d788bcb031'
        r['baseline_identity']=identity['aggregate_sha256']
        fixtures={}
        for item in protocol['fixtures']:
            path=ROOT/item['results'];assert sha(path)==item['sha256'];record=json.loads(path.read_text());assert record['status']=='completed' and record['passed'] and record['group']=='0' and all(x['outputs_exact'] and x['all_eight_states_exact'] for x in record['replays'])
            assert record['runtime_provenance']==r['runtime_provenance'] and record['loader_provenance']==milestone['compiled_identity']['loader_provenance']
            arrays={}
            for key,info in record['files'].items():
                file=path.parent/(key+'.npz');assert sha(file)==info['sha256']
                with np.load(file) as z:arrays[key]={k:z[k].copy() for k in z.files}
            validate_inputs(arrays['inputs']);prefix=record['prefix_length'];assert int(arrays['inputs']['positions'][0])==prefix
            assert len(arrays['state-before'])==8 and set(arrays['state-before'])==set(arrays['state-after'])
            fixtures[prefix]=arrays
        assert set(fixtures)=={32,128,512}
        # Reject out-of-domain calls before model construction or state creation.
        bad=[];original=fixtures[32]['inputs']
        for position in [0,31,2048]:
            x=dict(original);x['positions']=np.array([position],dtype=np.int32);bad.append(x)
        for name in ['x','ve','tap0','tap12','tap23']:
            x=dict(original);x[name]=np.repeat(x[name],2,axis=1);bad.append(x)
        x=dict(original);x['positions']=np.array([32,33],dtype=np.int32);bad.append(x)
        for x in bad:
            try:validate_inputs(x)
            except ValueError:pass
            else:raise AssertionError('Invalid scope accepted')
        r['scope_rejections']=len(bad)
        install()
        models={'A':ct.models.CompiledMLModel(str(baseline_root/'0.mlmodelc'),compute_units=ct.ComputeUnit.CPU_ONLY),'B':ct.models.CompiledMLModel(str(ROOT/build['compiled']),compute_units=ct.ComputeUnit.CPU_ONLY)}
        def call(prefix,arm):
            f=fixtures[prefix];validate_inputs(f['inputs']);model=models[arm];state=model.make_state()
            try:
                for n,v in f['state-before'].items():state.write_state(n,v)
                started=time.perf_counter_ns();got=model.predict(f['inputs'],state=state);elapsed=(time.perf_counter_ns()-started)/1e9
                outputs={n:bool(np.array_equal(got[n],v)) for n,v in f['outputs'].items()}
                states={n:bool(np.array_equal(state.read_state(n),v)) for n,v in f['state-after'].items()}
                exact=set(got)==set(f['outputs']) and all(outputs.values()) and all(states.values())
                row={'prefix':prefix,'arm':arm,'seconds':elapsed,'outputs_exact':outputs,'states_exact':states,'passed':exact}
                if not exact:
                    row['differences']={n:float(np.max(np.abs(got[n].astype(np.float64)-v))) for n,v in f['outputs'].items()};r['failed_call']=row;save();raise ValueError('Exact replay failed')
                return row
            finally:del state
        # All fixture parity gates must pass before ANY benchmark observations.
        for prefix in sorted(fixtures):
            for repeat in range(2):
                for arm in 'AB':r['correctness'].append(dict(call(prefix,arm),repeat=repeat))
            save();print('exact',prefix,flush=True)
        r['exactness_passed']=True;save()
        for prefix in sorted(fixtures):
            for arm in 'ABBAABBA':r['warmup'].append(call(prefix,arm))
        r['host_before']=host();save()
        for block,order in enumerate([[32,128,512],[512,32,128],[128,512,32]]):
            for prefix in order:
                for cycle in range(protocol['cycles_per_block']):
                    for slot,arm in enumerate('ABBA'):
                        r['timings'].append(dict(call(prefix,arm),block=block,cycle=cycle,slot=slot))
                save();print('timed',block,prefix,flush=True)
        r['host_after']=host()
        blocks=[]
        for block in range(3):
            for prefix in sorted(fixtures):
                rows=[x for x in r['timings'] if x['block']==block and x['prefix']==prefix]
                med={arm:statistics.median(x['seconds'] for x in rows if x['arm']==arm) for arm in 'AB'}
                reduction=1-med['B']/med['A'];blocks.append({'block':block,'prefix':prefix,'baseline_median_seconds':med['A'],'variant_median_seconds':med['B'],'median_reduction':reduction,'go':reduction>=protocol['go_median_reduction']})
        r['blocks']=blocks;r['performance_gate_passed']=all(x['go'] for x in blocks)
        r['prediction_guard']=stats()
        assert r['prediction_guard']=={'installed':True,'plain_calls':0,'state_calls':180,'gil_releases':180}
        models.clear()
        r['baseline_preservation']=verify_compiled_identity(baseline_root,ROOT/'evidence/05c-compiled/full-cache-1/compiled-identity.json',ROOT/'artifacts/04d-cpu/full-candidate-6',ROOT/'evidence/04d-cpu/full-candidate-6/package-identity.json')['aggregate_sha256']==r['baseline_identity']
        assert r['baseline_preservation'] and runtime_provenance()==r['runtime_provenance'] and sha(__file__)==r['script_sha256'] and sha(protocol_path)==r['protocol_sha256'] and inventory(ROOT/build['compiled'])==build['compiled_files']
        r.update(status='completed',experiment_completed=True,decision='local performance go; full integration still unqualified' if r['performance_gate_passed'] else 'no-go at predeclared threshold; do not integrate')
    except BaseException as e:r.update(status='failed',experiment_completed=False,error={'type':type(e).__name__,'message':str(e)});raise
    finally:save()
    print(json.dumps({'status':r['status'],'decision':r['decision'],'blocks':r['blocks']},indent=2))
if __name__=='__main__':main()
