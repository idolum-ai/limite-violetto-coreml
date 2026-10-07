"""Controlled original/candidate group0 replay on the localized failed prompt."""
from pathlib import Path
import argparse,gc,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
SOURCE=ROOT/'artifacts/04d-cpu/full-candidate-6';CACHE=ROOT/'artifacts/05c-compiled/full-cache-1'
FIXTURE=ROOT/'evidence/06c-gpu-localization/prompt0-1/results.json'
FIXTURE_SHA='ed27ff2a0b99bef14822450d044c2ec9e02206c647c934188c3f61c4c9886bb8'
SID='5a3e4c7a571e4fc3532377d9b9bb0da023f499c10709216fee7e1001bf4ad2c6'
CID='a838dd36e8f87c6b9fc0dd7865d0912651c6ab3399fa299d10ff12d788bcb031'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def inventory(p):return {str(x.relative_to(p)):sha(x) for x in sorted(p.rglob('*')) if x.is_file()}
def read(p):return json.loads(Path(p).read_text())
def save(p,r):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(r,indent=2,sort_keys=True,allow_nan=False)+'\n');tmp.replace(p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);p.add_argument('--build',required=True);a=p.parse_args()
 assert a.attempt and all(c in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in a.attempt)
 out=ROOT/'evidence/06d-gpu-rounding'/a.attempt;out.mkdir(parents=True,exist_ok=False)
 buildpath=ROOT/a.build
 assert buildpath.parent.parent==ROOT/'evidence/06d-gpu-rounding' and buildpath.name=='results.json'
 build_attempt=buildpath.parent.name;b=read(buildpath)
 assert b['package']==f'artifacts/06d-gpu-rounding/{build_attempt}/0.mlpackage' and b['compiled']==f'artifacts/06d-gpu-rounding/{build_attempt}/0.mlmodelc'
 assert sha(buildpath.parent/'source-snapshot.py')==b['script_sha256'] and sha(buildpath.parent/'helper-snapshot.py')==b['helper_sha256']
 assert b['passed'] and b['graph_check']['passed']
 assert b['source_candidate_identity']==SID and b['arithmetic_policy']=='explicit-signed-ties-even-v3-round-only' and b['selected_sites']=='all'
 assert b['graph_check']['replaced_round_count']==131 and b['graph_check']['remaining_round_count']==0
 assert b['status']=='completed' and b['helper_source']=='scripts/signed_even_round_v3.py' and b['builder_source']=='scripts/build_signed_even_group_v3.py'
 assert b['changed_package_files']==['Data/com.apple.CoreML/model.mlmodel']
 assert sha(ROOT/'scripts/signed_even_round_v3.py')==b['helper_sha256'] and sha(ROOT/'scripts/build_signed_even_group_v3.py')==b['script_sha256']
 operator_path=ROOT/'evidence/06d-gpu-rounding/round-replacement-1/results.json';operator=read(operator_path)
 assert operator['status']=='completed' and operator['bounded_replacement_gate_passed'] and operator['original_cpu_round_control_passed']
 assert operator['round_replacement_gate_passed'] and operator['native_cpu_full_q_preservation_passed'] and operator['helper_sha256']==b['helper_sha256']
 operator_sha=sha(operator_path)
 (out/'source-snapshot.py').write_bytes(Path(__file__).read_bytes());(out/'build-snapshot.json').write_bytes(buildpath.read_bytes())
 r={'status':'starting','script_sha256':sha(__file__),'build':a.build,'build_sha256':sha(buildpath),'fixture_sha256':FIXTURE_SHA,'operator_result':str(operator_path.relative_to(ROOT)),'operator_result_sha256':operator_sha,'strict_full_q_torch_passed':operator['strict_full_q_torch_passed'],'calls':[],'files':{},'actual_device_placement':'unverified; compute policy only','thresholds_changed':False}
 def update():save(out/'results.json',r)
 update();model=None
 try:
  import numpy as np
  import coremltools as ct
  from coreml_port.compiled_identity import verify_compiled_identity
  from coreml_port.prediction_guard import install,stats
  from scripts.runtime_provenance import runtime_provenance
  from scripts.probe_gpu_group0 import compare_array
  ident=verify_compiled_identity(CACHE,ROOT/'evidence/05c-compiled/full-cache-1/compiled-identity.json',SOURCE,ROOT/'evidence/04d-cpu/full-candidate-6/package-identity.json')
  assert ident['aggregate_sha256']==CID and ident['source_candidate_identity']==SID
  r['original_source_identity']=SID;r['original_compiled_identity']=CID
  assert inventory(ROOT/b['package'])==b['variant_files'] and inventory(ROOT/b['compiled'])==b['compiled_files']
  assert sha(FIXTURE)==FIXTURE_SHA;fixture=read(FIXTURE);r['runtime_provenance']=runtime_provenance();assert r['runtime_provenance']==fixture['runtime_provenance']
  states=ident['packages']['0']['state_names'];assert len(states)==8
  def archive(label,arrays):
   path=out/(label+'.npz');path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**arrays)
   info={'path':str(path.relative_to(ROOT)),'sha256':sha(path),'shapes':{n:list(x.shape) for n,x in arrays.items()},'dtypes':{n:str(x.dtype) for n,x in arrays.items()}};r['files'][label]=info;return info
  def load(label):
   info=fixture['files'][label];path=ROOT/info['path'];assert sha(path)==info['sha256']
   with np.load(path,allow_pickle=False) as z:return {n:z[n].copy() for n in z.files}
  def compare(actual,expected):
   assert set(actual)==set(expected)
   return {n:{**compare_array(expected[n],actual[n]),'bitwise_exact':actual[n].shape==expected[n].shape and actual[n].dtype==expected[n].dtype and actual[n].tobytes()==expected[n].tobytes()} for n in actual}
  def exact(metrics):return all(x['bitwise_exact'] for x in metrics.values())
  cases={}
  for c in (0,1):
   cases[c]={'inputs':load(f'chunk{c}/group0/own-CPU/inputs'),'before':load(f'chunk{c}/group0/own-CPU/state-before'),'CPU':{},'GPU':{}}
   for pol in ('CPU','GPU'):
    for kind in ('outputs','state-after'):cases[c][pol][kind]=load(f'chunk{c}/group0/replay-ICPU-SCPU-P{pol}/{kind}')
   assert set(cases[c]['before'])==set(states)
   for kind in ('inputs','before'):archive(f'chunk{c}/{kind}',cases[c][kind])
  install();assert stats()=={'installed':True,'plain_calls':0,'state_calls':0,'gil_releases':0}
  for arm,policy,path in [('original','CPU',CACHE/'0.mlmodelc'),('candidate','CPU',ROOT/b['compiled']),('candidate','GPU',ROOT/b['compiled'])]:
   model=ct.models.CompiledMLModel(str(path),compute_units=ct.ComputeUnit.CPU_ONLY if policy=='CPU' else ct.ComputeUnit.CPU_AND_GPU)
   for c in (0,1):
    case=cases[c];state=model.make_state()
    try:
     assert all(np.count_nonzero(state.read_state(n))==0 for n in states)
     for n,v in case['before'].items():state.write_state(n,v.copy())
     restored={n:state.read_state(n).copy() for n in states};assert exact(compare(restored,case['before']))
     payload={n:v.copy() for n,v in case['inputs'].items()};got={n:v.copy() for n,v in model.predict(payload,state=state).items()};after={n:state.read_state(n).copy() for n in states}
     assert exact(compare(payload,case['inputs'])) and all(np.isfinite(v).all() for d in (got,after) for v in d.values())
     row={'arm':arm,'policy':policy,'chunk':c,'restoration_exact':True,'outputs':archive(f'chunk{c}/{arm}-{policy}/outputs',got),'state_after':archive(f'chunk{c}/{arm}-{policy}/state-after',after),'vs_original_CPU':{'outputs':compare(got,case['CPU']['outputs']),'state':compare(after,case['CPU']['state-after'])}}
     row['vs_original_GPU']={'outputs':compare(got,case['GPU']['outputs']),'state':compare(after,case['GPU']['state-after'])}
     row['CPU_reference_bitwise_exact']=all(exact(x) for x in row['vs_original_CPU'].values());r['calls'].append(row);update()
     if policy=='CPU':assert row['CPU_reference_bitwise_exact'],f'CPU preservation failed for {arm} chunk{c}'
     print(arm,policy,c,'CPU exact',row['CPU_reference_bitwise_exact'],flush=True)
    finally:del state
   del model;model=None;gc.collect()
  assert len(r['calls'])==6 and stats()=={'installed':True,'plain_calls':0,'state_calls':6,'gil_releases':6}
  r['prediction_guard']=stats();r['cpu_preservation_passed']=True
  r['gpu_exact_gate_passed']=all(x['CPU_reference_bitwise_exact'] for x in r['calls'] if x['policy']=='GPU')
  assert sha(operator_path)==operator_sha
  assert sha(__file__)==r['script_sha256'] and sha(buildpath)==r['build_sha256'] and sha(FIXTURE)==FIXTURE_SHA
  assert sha(ROOT/'scripts/signed_even_round_v3.py')==b['helper_sha256'] and sha(ROOT/'scripts/build_signed_even_group_v3.py')==b['script_sha256']
  assert inventory(ROOT/b['package'])==b['variant_files'] and inventory(ROOT/b['compiled'])==b['compiled_files'] and runtime_provenance()==r['runtime_provenance']
  assert verify_compiled_identity(CACHE,ROOT/'evidence/05c-compiled/full-cache-1/compiled-identity.json',SOURCE,ROOT/'evidence/04d-cpu/full-candidate-6/package-identity.json')==ident
  r.update(status='completed',diagnostic_completed=True,passed=r['gpu_exact_gate_passed'],scope='Two shared CPU-input/cache prompt chunks; no full-model qualification.')
 except BaseException as e:r.update(status='failed',passed=False,error={'type':type(e).__name__,'message':str(e)});raise
 finally:model=None;gc.collect();update()
 print(json.dumps({'status':r['status'],'cpu_preservation_passed':r['cpu_preservation_passed'],'gpu_exact_gate_passed':r['gpu_exact_gate_passed']},indent=2))
if __name__=='__main__':main()
