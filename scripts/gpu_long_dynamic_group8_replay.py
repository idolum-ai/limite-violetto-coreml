"""Replay captured 1024:1026 group-8 input and state on dynamic-row package."""
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import coremltools as ct
import numpy as np
from coreml_port.cleanup_guard import install
install()
BASE=ROOT/'evidence/04d-cpu/long-group-locator'
OUT=ROOT/'evidence/04d-gpu/long-dynamic-group8-replay'
OUT.mkdir(parents=True,exist_ok=False)
input_path=BASE/'first-failure-inputs.npz';state_path=BASE/'first-failure-state-before.npz'
inp=np.load(input_path);before=np.load(state_path)
payload={k:inp[k] for k in ('x','ve','positions','tap0','tap12','tap23','norm_divisor')}
expected=inp['expected'];recorded_actual=inp['actual']
report={'compute_units':'CPU_ONLY','positions':payload['positions'].tolist(),
        'capture_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (input_path,state_path)},
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'state_names':sorted(before.files),'state_roundtrip_all_fp16_exact':all(np.array_equal(before[k],before[k].astype(np.float16).astype(np.float32)) for k in before.files),
        'experiments':[]}
def save():(OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
def metric(actual,target):
    d=np.abs(actual.astype(np.float64)-target.astype(np.float64));ix=np.argwhere(d!=0)
    return {'different_elements':int(len(ix)),'max_abs':float(d.max()),
            'rmse':float(np.sqrt(np.mean(d*d))),
            'first_differences':[{'index':i.tolist(),'actual':float(actual[tuple(i)]),'expected':float(target[tuple(i)])} for i in ix[:8]]}
save();arrays={'expected':expected,'recorded_actual':recorded_actual}
for name,package in (
    ('candidate3',ROOT/'artifacts/04d-cpu/full-candidate-3/8.mlpackage'),
    ('dynamic_raw',ROOT/'artifacts/04d-gpu/tiny3-dynamic-group8/8.mlpackage'),
):
    cm=ct.models.MLModel(str(package),compute_units=ct.ComputeUnit.CPU_ONLY)
    state=cm.make_state()
    write_dtypes={}
    for key in before.files:
        value=before[key].astype(np.float16)
        try:
            state.write_state(key,value)
            write_dtypes[key]='float16'
        except (RuntimeError,ValueError,TypeError):
            state.write_state(key,before[key])
            write_dtypes[key]='float32'
        assert np.array_equal(state.read_state(key).astype(np.float32),before[key]),key
    got=cm.predict(payload,state=state)
    arrays[name+'_y']=got['y'];arrays[name+'_tap23']=got['tap23_out']
    row={'name':name,'package':str(package.relative_to(ROOT)),
         'package_files_sha256':{str(p.relative_to(package)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(package.rglob('*')) if p.is_file()},
         'state_write_dtypes':write_dtypes,
         'y_vs_native':metric(got['y'],expected),
         'y_vs_recorded_candidate3':metric(got['y'],recorded_actual),
         'tap23_vs_input':metric(got['tap23_out'],payload['tap23'])}
    report['experiments'].append(row)
    print(name,'native',row['y_vs_native']['different_elements'],row['y_vs_native']['max_abs'],
          'recorded',row['y_vs_recorded_candidate3']['different_elements'],flush=True)
    save();np.savez_compressed(OUT/'outputs.npz',**arrays)
