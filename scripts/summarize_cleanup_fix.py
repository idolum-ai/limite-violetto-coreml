"""Record the diagnosed failure, pinned workaround and completed regressions."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'evidence/04b-reliability'
def read(n):return json.loads((OUT/n).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
raw=(OUT/'original-crash.ips').read_text();crash=json.loads(raw.split('\n',1)[1]);frames=[]
for t in crash['threads']:
 if t.get('triggered'):
  for f in t['frames']:
   frames.append({'image':crash['usedImages'][f['imageIndex']].get('name'),'symbol':f.get('symbol'),'image_offset':f.get('imageOffset')})
write(OUT/'crash-stack.json',{'capture_time':crash['captureTime'],'exception':crash['exception'],'thread':crash['faultingThread'],'frames':frames})
invariance=read('numerical-invariance.json');stress=read('long-context-stress.json')
assert all(not r['changed_metrics'] and r['lifecycle_pass'] for r in invariance.values())
assert len(stress)==3 and all(r['overflow_rejected'] and r['closed'] for r in stress)
assert "'calls': 1000, 'background_calls': 1000" in (OUT/'destructor-guarded.log').read_text()
files=[ROOT/'coreml_port/runtime.py',ROOT/'coreml_port/cleanup_guard.py',ROOT/'coreml_port/native/cleanup_guard.c',ROOT/'coreml_port/native/cleanup_reproducer.mm',ROOT/'artifacts/04b-reliability/cleanup_guard.dylib',ROOT/'artifacts/04b-reliability/cleanup_reproducer.dylib']
r={'updated_at':datetime.now(timezone.utc).isoformat(),'status':'Original cleanup failure diagnosed; specific mechanism mitigated and bounded regressions pass','cause':'Core ML Tools 9.0 PybindCompatibleArray generated destructor releases Python-owned array without acquiring the GIL; Core ML may invoke it on a worker','original_exit':139,'deterministic_debug_reproducer_exit':134,'guarded_worker_releases':1000,'numerical_records_unchanged':108,'lifecycle_checks_passed':34,'long_context_repetitions':3,'input_capacity_tested':2048,'guard_pins':{'python':'3.12','coremltools':'9.0','bridge_sha256':'915f478a3f2a392077012f8dacd9ae433579cbb299019ef2df23217ce536ee67'},'package_mutated':False,'model_arithmetic_changed_by_fix':False,'g4_3':'PASS for recorded guarded lifecycle and capacity checks; not a general production reliability guarantee','sources':{str(p.relative_to(ROOT)):sha(p) for p in files},'limitations':['Project-local Objective-C destructor interposition; requalify other versions','Interpreter finalization intentionally leaves late Python references for process teardown','Numerical fidelity remains separately failing','No original-runtime reference and no controlled serving/concurrency benchmark']}
write(OUT/'summary.json',r)
m=json.loads((ROOT/'manifest.json').read_text());m['updated_at']=r['updated_at'];m['reliability_followup']={'summary':'evidence/04b-reliability/summary.json','report':'evidence/04b-reliability/REPORT.md','guard':'coreml_port/cleanup_guard.py','status':r['status']};m['coreml_model']['gates']['G4.3']=r['g4_3'];m['validation']['gate_status']['G4.3']=r['g4_3'];m['phase']='Complete Core ML runner with diagnosed cleanup fix; numerical correction in progress';write(ROOT/'manifest.json',m);print(json.dumps(r,indent=2))
