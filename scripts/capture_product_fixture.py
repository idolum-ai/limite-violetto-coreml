"""Capture and replay one warm single-token group call from the qualified loader.

This is a correctness fixture, not a benchmark. It leaves all model/runtime
sources and compiled artifacts untouched and runs after the timing experiment.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--attempt',required=True);p.add_argument('--group',type=int,default=0);p.add_argument('--prefix',type=int,choices=[32,128,512],required=True);a=p.parse_args()
    if not 0<=a.group<12 or not a.attempt or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in a.attempt):p.error('Invalid group or attempt')
    out=ROOT/'evidence/05e-product'/a.attempt;out.mkdir(parents=True,exist_ok=False)
    (out/'source-snapshot.py').write_bytes(Path(__file__).read_bytes())
    r={'status':'starting','script_sha256':sha(__file__),'group':str(a.group),'prefix_length':a.prefix,'scope':'One warm single-token correctness capture/replay; no timing claim'}
    def save():
        t=out/'results.tmp';t.write_text(json.dumps(r,indent=2)+'\n');t.replace(out/'results.json')
    save()
    model=None;original=None;req=None
    try:
        import numpy as np
        import torch
        torch.set_num_threads(4)
        from coreml_port.loader_selection import create_pipeline,loader_provenance
        from scripts.runtime_provenance import runtime_provenance
        from coreml_port.prediction_guard import stats
        sel=argparse.Namespace(compute='CPU_ONLY',tiny_prefix_replay=True,
            model_dir=str(ROOT/'artifacts/04d-cpu/full-candidate-6'),
            identity_manifest=str(ROOT/'evidence/04d-cpu/full-candidate-6/package-identity.json'),
            compiled_dir=str(ROOT/'artifacts/05c-compiled/full-cache-1'),
            compiled_manifest=str(ROOT/'evidence/05c-compiled/full-cache-1/compiled-identity.json'))
        mp=ROOT/'evidence/05c-compiled/milestone-rerun1.json';m=json.loads(mp.read_text());r['milestone_sha256']=sha(mp)
        r['loader_provenance']=loader_provenance(sel);r['runtime_provenance']=runtime_provenance()
        if not m['milestone_pass'] or r['loader_provenance']!=m['compiled_identity']['loader_provenance'] or r['runtime_provenance']!=m['current_runtime_provenance']:
            raise ValueError('Qualified runtime identity differs')
        fixture=ROOT/'evidence/04-coreml/generation-comparison.json';ref=json.loads(fixture.read_text())['reference'];stream=ref['prompt_ids']+ref['generated_ids']
        scorepath=ROOT/'evidence/05-practical/coreml-pilot-1/scores.npz';pilotpath=scorepath.parent/'results.json'
        if sha(pilotpath)!='afff9529de181adf72be924ba09017eb3b7bf558362b153d0fad2c6f9b27d2e8':raise ValueError('Pilot identity differs')
        pilot=json.loads(pilotpath.read_text())
        if sha(fixture)!=pilot['fixture_sha256'] or sha(scorepath)!=pilot['scores_sha256']:raise ValueError('Fixture identity differs')
        r.update(fixture_sha256=sha(fixture),score_fixture_sha256=sha(scorepath),prompt_ids=stream[:a.prefix],next_token=stream[a.prefix])
        model=create_pipeline(sel);original=model.predict;req=model.request()
        with np.load(scorepath) as z:
            first=model.forward(stream[:a.prefix],req);r['prefill_scores_exact']=bool(np.array_equal(first,z[f'p{a.prefix}_s0']))
            captured={};group=str(a.group)
            state_names=model.compiled_identity['packages'][group]['state_names'];r['state_names']=state_names
            def capture(name,inputs,state=None):
                if name!=group:return original(name,inputs,state)
                if captured:raise ValueError('More than one group call during single-token decode')
                actual=dict(inputs)
                if name in model.norm_divisor_inputs:actual['norm_divisor']=np.array([1280.],dtype=np.float32)
                captured['inputs']={k:v.copy() for k,v in actual.items()}
                captured['state-before']={n:state.read_state(n).copy() for n in state_names}
                result=original(name,inputs,state)
                captured['outputs']={k:v.copy() for k,v in result.items()}
                captured['state-after']={n:state.read_state(n).copy() for n in state_names}
                return result
            model.predict=capture
            second=model.forward([stream[a.prefix]],req);model.predict=original
            r['decode_scores_exact']=bool(np.array_equal(second,z[f'p{a.prefix}_s1']))
        if set(captured)!= {'inputs','outputs','state-before','state-after'}:raise ValueError('Capture incomplete')
        req.close();r['request_disposed']=req.closed and not req.states and not req.warmup_ids;req=None
        r['files']={}
        for name,values in captured.items():
            path=out/(name+'.npz');np.savez(path,**values)
            r['files'][name]={'sha256':sha(path),'arrays':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in values.items()}}
        r['replays']=[]
        for repeat in range(2):
            state=model.models[group].make_state()
            try:
                for n,v in captured['state-before'].items():state.write_state(n,v)
                actual=model.models[group].predict(captured['inputs'],state=state)
                outputs_exact=set(actual)==set(captured['outputs']) and all(np.array_equal(actual[n],v) for n,v in captured['outputs'].items())
                states_exact=all(np.array_equal(state.read_state(n),v) for n,v in captured['state-after'].items())
                r['replays'].append({'repeat':repeat,'outputs_exact':bool(outputs_exact),'all_eight_states_exact':bool(states_exact)})
            finally:del state
        r['prediction_guard']=stats()
        r['passed']=r['prefill_scores_exact'] and r['decode_scores_exact'] and r['request_disposed'] and len(state_names)==8 and all(x['outputs_exact'] and x['all_eight_states_exact'] for x in r['replays'])
        if not r['passed']:raise ValueError('Capture/replay parity failed')
        if sha(__file__)!=r['script_sha256'] or loader_provenance(sel)!=r['loader_provenance']:raise ValueError('Source drift')
        r['status']='completed'
    except BaseException as exc:
        r.update(status='failed',passed=False,failure={'type':type(exc).__name__,'message':str(exc)});raise
    finally:
        if model is not None and original is not None:model.predict=original
        if req is not None:req.close()
        save()
    print(json.dumps({'passed':r['passed'],'replays':r['replays']},indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
