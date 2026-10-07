"""Reproduce the prediction/reset GIL wait with disposable child processes."""
import argparse,ctypes,hashlib,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
LIB=ROOT/'artifacts/04e-runtime/prediction_guard.dylib'
OUT=ROOT/'evidence/04e-runtime/prediction-guard-synthetic.json'
p=argparse.ArgumentParser();p.add_argument('--child',choices=['holding','released']);args=p.parse_args()
if args.child:
    lib=ctypes.PyDLL(str(LIB))
    lib.violetto_synthetic_prediction.argtypes=[ctypes.c_int]
    lib.violetto_synthetic_prediction.restype=ctypes.c_int
    code=lib.violetto_synthetic_prediction(int(args.child=='released'))
    lib.violetto_prediction_synthetic_workers.restype=ctypes.c_ulong
    workers=lib.violetto_prediction_synthetic_workers()
    print(json.dumps({'mode':args.child,'return_code':code,'workers':workers}),flush=True)
    raise SystemExit(code or workers!=1)

rows=[]
for mode,seconds in [('holding',3),('released',6)]:
    command=[sys.executable,str(Path(__file__).resolve()),'--child',mode]
    began=time.monotonic()
    try:
        result=subprocess.run(command,capture_output=True,text=True,timeout=seconds,check=False)
        row={'mode':mode,'timed_out':False,'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
    except subprocess.TimeoutExpired as exc:
        row={'mode':mode,'timed_out':True,'exit_code':None,
             'stdout':exc.stdout.decode(errors='replace') if isinstance(exc.stdout,bytes) else exc.stdout,
             'stderr':exc.stderr.decode(errors='replace') if isinstance(exc.stderr,bytes) else exc.stderr}
    row['seconds']=time.monotonic()-began;rows.append(row);print(mode,row['timed_out'],row['exit_code'],flush=True)
passed=rows[0]['timed_out'] and not rows[1]['timed_out'] and rows[1]['exit_code']==0
digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
record={'status':'complete','passed':passed,'expectation':'GIL-held synthetic child times out; released child completes',
        'script_sha256':digest(Path(__file__)),
        'guard_source_sha256':digest(ROOT/'coreml_port/native/prediction_guard.mm'),
        'guard_binary_sha256':digest(LIB),'rows':rows}
OUT.write_text(json.dumps(record,indent=2)+'\n')
raise SystemExit(0 if passed else 1)
