"""Native all-domain group0 parity for a lazy short/full score conditional."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def inventory(p):return {str(f.relative_to(p)):sha(f) for f in sorted(p.rglob('*')) if f.is_file()}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',required=True);p.add_argument('--attempt',required=True);a=p.parse_args()
    if not a.attempt or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in a.attempt):p.error('Invalid attempt')
    out=ROOT/'evidence/05g-integration'/a.attempt;out.mkdir(parents=True,exist_ok=False);(out/'source-snapshot.py').write_bytes(Path(__file__).read_bytes())
    buildpath=ROOT/a.build;build=json.loads(buildpath.read_text());r={'status':'starting','script_sha256':sha(__file__),'build_sha256':sha(buildpath),'build':a.build,'comparisons':[],'state_sequences':[]}
    def save():
        tmp=out/'results.tmp';tmp.write_text(json.dumps(r,indent=2)+'\n');tmp.replace(out/'results.json')
    save()
    try:
        import coremltools as ct
        import numpy as np
        from coreml_port.prediction_guard import install,stats
        from coreml_port.compiled_identity import verify_compiled_identity
        from scripts.runtime_provenance import runtime_provenance
        baseline=ROOT/'artifacts/05c-compiled/full-cache-1';source=ROOT/'artifacts/04d-cpu/full-candidate-6'
        identity=verify_compiled_identity(baseline,ROOT/'evidence/05c-compiled/full-cache-1/compiled-identity.json',source,ROOT/'evidence/04d-cpu/full-candidate-6/package-identity.json')
        assert identity['aggregate_sha256']=='a838dd36e8f87c6b9fc0dd7865d0912651c6ab3399fa299d10ff12d788bcb031'
        r['baseline_identity']=identity['aggregate_sha256'];r['runtime_provenance']=runtime_provenance()
        assert r['runtime_provenance']==json.loads((ROOT/'evidence/05c-compiled/milestone-rerun1.json').read_text())['current_runtime_provenance']
        assert build['passed'] and build['graph_check']['passed'] and inventory(ROOT/build['compiled'])==build['compiled_files'] and inventory(ROOT/build['package'])==build['variant_files']
        fixture=ROOT/'evidence/04-coreml/generation-comparison.json';assert sha(fixture)=='1ee6553885123a57b8fb40b89c292eea201e148e6f3bd4d9b4cba1c057f9f3eb';ref=json.loads(fixture.read_text())['reference'];stream=ref['prompt_ids']+ref['generated_ids']
        names=identity['packages']['0']['state_names'];assert len(names)==8
        install();models={arm:ct.models.CompiledMLModel(str(path),compute_units=ct.ComputeUnit.CPU_ONLY) for arm,path in [('A',baseline/'0.mlmodelc'),('B',ROOT/build['compiled'])]}
        embedding=ct.models.CompiledMLModel(str(baseline/'embedding.mlmodelc'),compute_units=ct.ComputeUnit.CPU_ONLY)
        plain_calls=state_calls=0
        def inputs(start,count):
            nonlocal plain_calls
            e=embedding.predict({'ids':np.array([stream[start:start+count]],np.int32),'norm_divisor':np.array([1280.],np.float32)});plain_calls+=1
            return {'x':e['x'],'ve':e['ve'],'positions':np.arange(start,start+count,dtype=np.int32),'tap0':e['x'].copy(),'tap12':e['x'].copy(),'tap23':e['x'].copy(),'norm_divisor':np.array([1280.],np.float32)}
        def fresh():
            states={arm:m.make_state() for arm,m in models.items()}
            assert all(np.count_nonzero(s.read_state(n))==0 for s in states.values() for n in names)
            return states
        def compare(label,payload,states,expected=None):
            nonlocal state_calls
            got={};after={}
            for arm in 'AB':
                got[arm]=models[arm].predict(payload,state=states[arm]);state_calls+=1
                after[arm]={n:states[arm].read_state(n).copy() for n in names}
            outputs={n:bool(np.array_equal(got['A'][n],got['B'][n])) for n in ['y','tap23_out']}
            caches={n:bool(np.array_equal(after['A'][n],after['B'][n])) for n in names}
            sealed=True if expected is None else all(np.array_equal(got[arm][n],v) for arm in 'AB' for n,v in expected['outputs'].items()) and all(np.array_equal(after[arm][n],v) for arm in 'AB' for n,v in expected['state-after'].items())
            row={'label':label,'positions':payload['positions'].tolist(),'outputs_exact':outputs,'states_exact':caches,'sealed_fixture_exact':bool(sealed),'passed':all(outputs.values()) and all(caches.values()) and sealed};r['comparisons'].append(row);save()
            if not row['passed']:
                for arm in 'AB':np.savez(out/f'failure-{arm}-outputs.npz',**got[arm]);np.savez(out/f'failure-{arm}-states.npz',**after[arm])
                raise ValueError(f'Parity failed: {label}')
        for count in range(1,33):
            states=fresh()
            try:compare(f'cold-{count}',inputs(0,count),states)
            finally:states.clear()
        print('cold1..32 exact',flush=True)
        for i,chunks in enumerate([[3,1,11,1,16,1],[1,1,13,1,16,1],[32,32,1]]):
            states=fresh();position=0
            try:
                for j,count in enumerate(chunks):compare(f'sequence-{i}-step-{j}',inputs(position,count),states);position+=count
            finally:states.clear()
            r['state_sequences'].append({'chunks':chunks,'final_position':position,'states_disposed':not states});save()
        print('persistent-state boundary sequences exact',flush=True)
        protocol=ROOT/'evidence/05e-product/protocol.json';assert sha(protocol)=='9d9e0656b224975d27551ca489f3448c69d62bc7401c4cef48d232f7b50a6606'
        for item in json.loads(protocol.read_text())['fixtures']:
            path=ROOT/item['results'];assert sha(path)==item['sha256'];record=json.loads(path.read_text());arrays={}
            for kind,info in record['files'].items():
                file=path.parent/(kind+'.npz');assert sha(file)==info['sha256']
                with np.load(file) as z:arrays[kind]={n:z[n].copy() for n in z.files}
            states=fresh()
            try:
                for s in states.values():
                    for n,v in arrays['state-before'].items():s.write_state(n,v)
                compare(f"warm-{record['prefix_length']}",arrays['inputs'],states,arrays)
            finally:states.clear()
        r['prediction_guard']=stats();assert r['prediction_guard']=={'installed':True,'plain_calls':plain_calls,'state_calls':state_calls,'gil_releases':plain_calls+state_calls}
        assert len(r['comparisons'])==50 and plain_calls==47 and state_calls==100
        models.clear();del embedding
        assert sha(__file__)==r['script_sha256'] and sha(buildpath)==r['build_sha256'] and runtime_provenance()==r['runtime_provenance'] and inventory(ROOT/build['compiled'])==build['compiled_files']
        r.update(status='completed',passed=True,comparison_count=50,scope='Original interface all cold counts1..32, persistent short/full transitions, sealed warm positions32/128/512')
    except BaseException as e:r.update(status='failed',passed=False,error={'type':type(e).__name__,'message':str(e)});raise
    finally:save()
    print(json.dumps({'passed':r['passed'],'comparisons':50,'guard':r['prediction_guard']},indent=2))
if __name__=='__main__':main()
