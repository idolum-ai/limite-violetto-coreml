"""Own-process, two-prediction group-0 replay worker for an external trace.

This script does not record or attach a profiler. CPU/GPU are compute-policy
labels; actual placement remains unverified until the own-PID trace is parsed.
No reference model or full pipeline is constructed.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
FIXTURE=ROOT/'evidence/06c-gpu-localization/prompt0-1/results.json'
FIXTURE_SHA256='ed27ff2a0b99bef14822450d044c2ec9e02206c647c934188c3f61c4c9886bb8'
SOURCE=ROOT/'artifacts/04d-cpu/full-candidate-6'
CACHE=ROOT/'artifacts/05c-compiled/full-cache-1'
SOURCE_MANIFEST=ROOT/'evidence/04d-cpu/full-candidate-6/package-identity.json'
COMPILED_MANIFEST=ROOT/'evidence/05c-compiled/full-cache-1/compiled-identity.json'
SOURCE_ID='5a3e4c7a571e4fc3532377d9b9bb0da023f499c10709216fee7e1001bf4ad2c6'
COMPILED_ID='a838dd36e8f87c6b9fc0dd7865d0912651c6ab3399fa299d10ff12d788bcb031'
STATE_NAMES={f'inner_{kind}_cache_{i}' for kind in ('k','v') for i in range(4)}
INPUT_NAMES={'x','ve','positions','tap0','tap12','tap23','norm_divisor'}
OUTPUT_NAMES={'y','tap23_out'}


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def stamp():
    return {'wall_time_ns':time.time_ns(),'monotonic_ns':time.monotonic_ns(),'perf_counter_ns':time.perf_counter_ns()}


def read(path):
    value=json.loads(Path(path).read_text())
    if not isinstance(value,dict):raise ValueError(f'Expected JSON object: {path}')
    return value


def save(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');temp.replace(path)


def pinned_fixture():
    if sha(FIXTURE)!=FIXTURE_SHA256:raise ValueError('Completed source replay result identity changed')
    r=read(FIXTURE)
    if r.get('status')!='completed' or r.get('passed') is not True or r.get('diagnostic_completed') is not True:
        raise ValueError('Source replay did not complete its preservation checks')
    if r.get('source_identity')!=SOURCE_ID or r.get('compiled_identity')!=COMPILED_ID or r.get('thresholds_changed') is not False:
        raise ValueError('Source replay cohort or thresholds differ')
    if sha(FIXTURE.parent/'source-snapshot.py')!=r['script_sha256'] or sha(FIXTURE.parent/'protocol.json')!=r['protocol_sha256']:
        raise ValueError('Source replay script/protocol snapshot differs')
    for name,digest in r['qualified_sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError(f'Qualified source changed: {name}')
    return r


def selected_archives(policy):
    if policy not in {'CPU','GPU'}:raise ValueError('Expected CPU or GPU policy')
    result=[]
    for chunk in range(2):
        stem=f'chunk{chunk}/group0'
        result.append({'chunk':chunk,'inputs':f'{stem}/own-CPU/inputs','state_before':f'{stem}/own-CPU/state-before',
                       'expected_outputs':f'{stem}/replay-ICPU-SCPU-P{policy}/outputs',
                       'expected_state_after':f'{stem}/replay-ICPU-SCPU-P{policy}/state-after'})
    return result


def load_archive(record,label):
    import numpy as np
    info=record['files'][label];path=(ROOT/info['path']).resolve()
    if not path.is_relative_to(FIXTURE.parent.resolve()) or sha(path)!=info['sha256']:
        raise ValueError(f'Archive path/digest differs: {label}')
    with np.load(path,allow_pickle=False) as data:arrays={key:data[key].copy() for key in data.files}
    if {k:list(v.shape) for k,v in arrays.items()}!=info['shapes'] or {k:str(v.dtype) for k,v in arrays.items()}!=info['dtypes']:
        raise ValueError(f'Archive shape/dtype inventory differs: {label}')
    if any(not np.isfinite(v).all() for v in arrays.values()):raise ValueError(f'Nonfinite archive: {label}')
    return arrays,{'label':label,**info}


def validate_arrays(chunk,case):
    import numpy as np
    start,count=((0,32),(32,12))[chunk]
    inputs=case['inputs'];before=case['state_before'];outputs=case['expected_outputs'];after=case['expected_state_after']
    if set(inputs)!=INPUT_NAMES or set(before)!=STATE_NAMES or set(after)!=STATE_NAMES or set(outputs)!=OUTPUT_NAMES:
        raise ValueError('Unexpected group-0 input/state/output names')
    shapes={**{n:(1,count,1280) for n in ('x','tap0','tap12','tap23')},'ve':(1,count,2,128),'positions':(count,),'norm_divisor':(1,)}
    for name,value in inputs.items():
        if value.shape!=shapes[name] or value.dtype!=(np.int32 if name=='positions' else np.float32):
            raise ValueError(f'Invalid effective input: {name}')
    if not np.array_equal(inputs['positions'],np.arange(start,start+count,dtype=np.int32)) or not np.array_equal(inputs['norm_divisor'],np.array([1280.],np.float32)):
        raise ValueError('Frozen positions or norm divisor differs')
    if any(v.shape!=(1,2048,2,128) or v.dtype!=np.float32 for d in (before,after) for v in d.values()):
        raise ValueError('Unexpected state API array shape/dtype')
    if any(v.shape!=(1,count,1280) or v.dtype!=np.float32 for v in outputs.values()):
        raise ValueError('Unexpected saved output shape/dtype')
    if chunk==0 and any(np.count_nonzero(v) for v in before.values()):raise ValueError('Cold state is not zero')


def compare_maps(actual,expected):
    import numpy as np
    result={'names_exact':set(actual)==set(expected),'arrays':{}}
    for name in sorted(set(actual)&set(expected)):
        a,b=actual[name],expected[name];shape=a.shape==b.shape;dtype=a.dtype==b.dtype
        exact=shape and dtype and bool(np.array_equal(a,b))
        row={'shape_exact':shape,'dtype_exact':dtype,'exact':exact,
             'bitwise_exact':shape and dtype and a.tobytes(order='C')==b.tobytes(order='C')}
        if shape and not exact:
            row['different_elements']=int(np.count_nonzero(a!=b))
            row['max_abs']=float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64))))
            coords=np.argwhere(a!=b)
            row['first_difference']=coords[0].tolist() if len(coords) else None
        result['arrays'][name]=row
    result['exact']=result['names_exact'] and all(v['exact'] for v in result['arrays'].values())
    result['bitwise_exact']=result['names_exact'] and all(v['bitwise_exact'] for v in result['arrays'].values())
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--policy',choices=['CPU','GPU'],required=True);p.add_argument('--attempt',required=True)
    args=p.parse_args()
    if not args.attempt or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in args.attempt):p.error('Fresh path-safe lowercase attempt required')
    out=ROOT/'evidence/06c-gpu-localization'/args.attempt;out.mkdir(parents=True,exist_ok=False)
    (out/'source-snapshot.py').write_bytes(Path(__file__).read_bytes())
    record={'status':'starting','pid':os.getpid(),'policy':args.policy,'compute':'CPU_ONLY' if args.policy=='CPU' else 'CPU_AND_GPU',
            'group':'0','scope':'Two independent original group-0 predictions, CPU-trajectory inputs/cache, chunks32 and12',
            'actual_device_placement':'unverified; external own-PID execution trace required',
            'script_sha256':sha(__file__),'source_result':str(FIXTURE.relative_to(ROOT)),'source_result_sha256':FIXTURE_SHA256,
            'start':stamp(),'archives':[],'calls':[],'files':{},'thresholds_changed':False}
    def update():save(out/'results.json',record)
    def event(kind,**fields):
        row={'event':kind,'pid':os.getpid(),'policy':args.policy,**stamp(),**fields}
        with (out/'events.jsonl').open('a') as stream:stream.write(json.dumps(row,sort_keys=True)+'\n')
        print(json.dumps(row,sort_keys=True),flush=True)
        return row
    update();event('worker_start');model=None;state=None
    try:
        import numpy as np
        source=pinned_fixture();record['qualified_sources']=source['qualified_sources']
        cases=[]
        for choice in selected_archives(args.policy):
            case={'chunk':choice['chunk']}
            for key,label in choice.items():
                if key=='chunk':continue
                arrays,info=load_archive(source,label);case[key]=arrays;record['archives'].append(info)
            validate_arrays(choice['chunk'],case);cases.append(case)
        from scripts.runtime_provenance import runtime_provenance
        record['runtime_provenance']=runtime_provenance()
        if record['runtime_provenance']!=source['runtime_provenance']:raise ValueError('Runtime/guard provenance differs from saved replay')
        from coreml_port.prediction_guard import install,stats
        from coreml_port.cleanup_guard import stats as cleanup_stats
        install();record['guard_before']=stats();record['cleanup_guard_before']=cleanup_stats()
        if record['guard_before']!={'installed':True,'plain_calls':0,'state_calls':0,'gil_releases':0}:raise ValueError('Worker prediction counters were not fresh')
        import coremltools as ct
        from coreml_port.compiled_identity import verify_compiled_identity
        event('identity_verification_begin')
        identity=verify_compiled_identity(CACHE,COMPILED_MANIFEST,SOURCE,SOURCE_MANIFEST)
        if identity['aggregate_sha256']!=COMPILED_ID or identity['source_candidate_identity']!=SOURCE_ID:raise ValueError('Original source/cache identity differs')
        record.update(source_identity=SOURCE_ID,compiled_identity=COMPILED_ID,
                      source_manifest_sha256=sha(SOURCE_MANIFEST),compiled_manifest_sha256=sha(COMPILED_MANIFEST),
                      source_root=str(SOURCE),compiled_root=str(CACHE))
        group=identity['packages']['0']
        if set(group['state_names'])!=STATE_NAMES or set(group['input_names'])!=INPUT_NAMES or set(group['output_names'])!=OUTPUT_NAMES:
            raise ValueError('Original group schema differs from effective saved replay')
        event('identity_verification_end');update()
        event('model_load_begin');start=time.perf_counter()
        model=ct.models.CompiledMLModel(str(CACHE/'0.mlmodelc'),compute_units=ct.ComputeUnit.CPU_ONLY if args.policy=='CPU' else ct.ComputeUnit.CPU_AND_GPU)
        record['load_seconds']=time.perf_counter()-start;event('model_load_end');update()
        for case in cases:
            chunk=case['chunk'];row={'chunk':chunk,'positions':case['inputs']['positions'].tolist(),'expected_replay':f'ICPU-SCPU-P{args.policy}','state_reference_released':False}
            record['calls'].append(row)
            try:
                state=model.make_state()
                for name,value in case['state_before'].items():state.write_state(name,value)
                restored={name:state.read_state(name).copy() for name in sorted(STATE_NAMES)}
                row['restoration']=compare_maps(restored,case['state_before'])
                if not row['restoration']['exact']:raise ValueError('Incoming state restoration differs from saved CPU trajectory')
                payload={name:value.copy() for name,value in case['inputs'].items()}
                update();row['prediction_begin']=event('prediction_begin',chunk=chunk,call_ordinal=chunk+1)
                start=time.perf_counter_ns();got=model.predict(payload,state=state);stop=time.perf_counter_ns()
                row['prediction_end']=event('prediction_end',chunk=chunk,call_ordinal=chunk+1)
                row['predict_seconds']=(stop-start)/1e9
                outputs={name:value.copy() for name,value in got.items()};after={name:state.read_state(name).copy() for name in sorted(STATE_NAMES)}
                row['inputs_unchanged']=compare_maps(payload,case['inputs'])
                row['outputs']=compare_maps(outputs,case['expected_outputs']);row['state_after']=compare_maps(after,case['expected_state_after'])
                for kind,arrays in [('outputs',outputs),('state-after',after)]:
                    path=out/f'chunk{chunk}-{kind}.npz';np.savez_compressed(path,**arrays)
                    record['files'][path.name]={'sha256':sha(path),'shapes':{n:list(v.shape) for n,v in arrays.items()},'dtypes':{n:str(v.dtype) for n,v in arrays.items()}}
                row['passed']=all(row[key]['exact'] for key in ('inputs_unchanged','outputs','state_after'))
                update()
                if not row['passed']:raise ValueError(f'Chunk {chunk} differs from saved replay under the same compute policy')
            finally:
                state=None;gc.collect();row['state_reference_released']=True;row['release_time']=stamp();update()
        record['prediction_guard']=stats();record['cleanup_guard_after']=cleanup_stats()
        if record['prediction_guard']!={'installed':True,'plain_calls':0,'state_calls':2,'gil_releases':2}:raise ValueError('Expected exactly two guarded state predictions')
        for info in record['archives']:
            if sha(ROOT/info['path'])!=info['sha256']:raise ValueError('Loaded fixture archive changed during replay')
        if pinned_fixture()!=source or runtime_provenance()!=record['runtime_provenance'] or sha(__file__)!=record['script_sha256']:
            raise ValueError('Source replay, runtime or worker source changed')
        if verify_compiled_identity(CACHE,COMPILED_MANIFEST,SOURCE,SOURCE_MANIFEST)!=identity:
            raise ValueError('Original source/cache changed during replay')
        record.update(status='completed',passed=True,fixture_preserved=True,source_and_compiled_preserved=True,
                      owned_state_references_released=all(c['state_reference_released'] for c in record['calls']))
    except BaseException as exc:
        record.update(status='failed',passed=False,error={'type':type(exc).__name__,'message':str(exc)})
        event('worker_failure',error_type=type(exc).__name__,message=str(exc));raise
    finally:
        state=None;model=None;gc.collect();record['finish']=stamp();update();event('worker_finish',status=record['status'],passed=record.get('passed',False))
    return 0


if __name__=='__main__':raise SystemExit(main())
