"""Twelve restored-state calls for a one-site reciprocal candidate and tap controls."""
from pathlib import Path
import argparse,gc,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
BASE=ROOT/'evidence/06d-gpu-rounding/group0-1/results.json'
PRIOR=ROOT/'evidence/06d-gpu-rounding/group0-replay-1/results.json'
TAP_PRIOR=ROOT/'evidence/06e-gpu-normalization/replay-1/results.json'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def inventory(p):return {str(x.relative_to(p)):sha(x) for x in sorted(p.rglob('*')) if x.is_file()}
def save(p,r):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(r,indent=2,sort_keys=True,allow_nan=False)+'\n');t.replace(p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--build',required=True);p.add_argument('--attempt',required=True);p.add_argument('--protocol',required=True);a=p.parse_args()
 assert a.attempt and all(c in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in a.attempt)
 pp=ROOT/a.protocol;protocol=read(pp);assert protocol['attempt']==a.attempt and protocol['build']==a.build
 assert {str(Path(__file__).relative_to(ROOT)),a.build,str(BASE.relative_to(ROOT)),str(PRIOR.relative_to(ROOT)),str(TAP_PRIOR.relative_to(ROOT))} <= set(protocol['pins'])
 assert all(sha(ROOT/n)==h for n,h in protocol['pins'].items())
 bp=ROOT/a.build;b=read(bp);base=read(BASE);prior=read(PRIOR);tap_prior=read(TAP_PRIOR)
 assert b['status']=='completed' and b['passed'] and b['graph_check']['passed']
 assert base['passed'] and prior['status']=='completed' and prior['cpu_preservation_passed'] and tap_prior['instrumentation_valid']
 out=ROOT/'evidence/06f-gpu-reciprocal'/a.attempt;out.mkdir(parents=True,exist_ok=False)
 for name,path in [('source-snapshot.py',Path(__file__)),('build-snapshot.json',bp),('protocol.json',pp)]: (out/name).write_bytes(path.read_bytes())
 r={'status':'starting','pins':protocol['pins'],'protocol_sha256':sha(pp),'files':{},'calls':[],'controls':[],'actual_device_placement':'unverified; compute policy only'}
 def update():save(out/'results.json',r)
 update();model=None
 try:
  import coremltools as ct
  import numpy as np
  from coreml_port.prediction_guard import install,stats
  from scripts.runtime_provenance import runtime_provenance
  from scripts.probe_gpu_group0 import compare_array
  from scripts.probe_reciprocal_refinement import oracle
  assert inventory(ROOT/base['package'])==base['variant_files'] and inventory(ROOT/base['compiled'])==base['compiled_files']
  for key,inv in [('package','variant_files'),('compiled','compiled_files'),('tapped_package','tapped_variant_files'),('tapped_compiled','tapped_compiled_files')]:assert inventory(ROOT/b[key])==b[inv],key
  assert b['source_result_sha256']==sha(BASE) and b['source_files']==base['variant_files']
  seed=b['tapped_graph_check']['seed_name'];taps=tap_prior['taps']+[seed];r['taps']=taps
  r['runtime_provenance']=runtime_provenance();assert r['runtime_provenance']==prior['runtime_provenance']==tap_prior['runtime_provenance']
  names=[f'inner_{k}_cache_{i}' for i in range(4) for k in ('k','v')]
  def load(rec,label):
   info=rec['files'][label];path=ROOT/info['path'];assert sha(path)==info['sha256']
   with np.load(path,allow_pickle=False) as z:return {n:z[n].copy() for n in z.files}
  def archive(label,arrays):
   path=out/(label+'.npz');path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**arrays)
   info={'path':str(path.relative_to(ROOT)),'sha256':sha(path),'shapes':{n:list(v.shape) for n,v in arrays.items()},'dtypes':{n:str(v.dtype) for n,v in arrays.items()}};r['files'][label]=info;return info
  def compare(actual,expected):
   assert set(actual)==set(expected)
   return {n:{**compare_array(expected[n],actual[n]),'bitwise_exact':actual[n].shape==expected[n].shape and actual[n].dtype==expected[n].dtype and actual[n].tobytes()==expected[n].tobytes()} for n in actual}
  def exact(c):return all(v['bitwise_exact'] for v in c.values())
  cases={c:{'inputs':load(prior,f'chunk{c}/inputs'),'before':load(prior,f'chunk{c}/before'),**{pol:{'outputs':load(prior,f'chunk{c}/candidate-{pol}/outputs'),'state':load(prior,f'chunk{c}/candidate-{pol}/state-after'),'taps':load(tap_prior,f'chunk{c}/{pol}-tapped/outputs')} for pol in ('CPU','GPU')}} for c in (0,1)}
  install();assert stats()=={'installed':True,'plain_calls':0,'state_calls':0,'gil_releases':0}
  got={}
  for pol in ('CPU','GPU'):
   for arm,compiled in [('original',base['compiled']),('candidate',b['compiled']),('tapped',b['tapped_compiled'])]:
    model=ct.models.CompiledMLModel(str(ROOT/compiled),compute_units=ct.ComputeUnit.CPU_ONLY if pol=='CPU' else ct.ComputeUnit.CPU_AND_GPU)
    for c,case in cases.items():
     state=model.make_state()
     try:
      assert all(np.count_nonzero(state.read_state(n))==0 for n in names)
      for n,v in case['before'].items():state.write_state(n,v.copy())
      assert exact(compare({n:state.read_state(n).copy() for n in names},case['before']))
      payload={n:v.copy() for n,v in case['inputs'].items()};outputs={n:v.copy() for n,v in model.predict(payload,state=state).items()};post={n:state.read_state(n).copy() for n in names}
      assert exact(compare(payload,case['inputs'])) and all(np.isfinite(v).all() for d in (outputs,post) for v in d.values())
      assert set(outputs)==({'y','tap23_out'}|set(taps) if arm=='tapped' else {'y','tap23_out'})
      got[(pol,arm,c)]={'outputs':outputs,'state':post};label=f'chunk{c}/{pol}-{arm}'
      r['calls'].append({'policy':pol,'arm':arm,'chunk':c,'outputs':archive(label+'/outputs',outputs),'state_after':archive(label+'/state-after',post),'restoration_exact':True});update()
     finally:del state
    del model;model=None;gc.collect()
  for pol in ('CPU','GPU'):
   for c in (0,1):
    original=got[(pol,'original',c)];candidate=got[(pol,'candidate',c)];tapped=got[(pol,'tapped',c)]
    control={'policy':pol,'chunk':c,'original_outputs_vs_saved':compare(original['outputs'],cases[c][pol]['outputs']),'original_states_vs_saved':compare(original['state'],cases[c][pol]['state']),'tapped_endpoints_vs_candidate':compare({n:tapped['outputs'][n] for n in ('y','tap23_out')},candidate['outputs']),'tapped_states_vs_candidate':compare(tapped['state'],candidate['state'])}
    control['passed']=all(exact(control[n]) for n in ('original_outputs_vs_saved','original_states_vs_saved','tapped_endpoints_vs_candidate','tapped_states_vs_candidate'));r['controls'].append(control)
  r['instrumentation_valid']=all(x['passed'] for x in r['controls']);r['cpu_preservation']=[];r['gpu_vs_original_cpu']=[]
  for c in (0,1):
   original=got[('CPU','original',c)];candidate=got[('CPU','candidate',c)];gpu=got[('GPU','candidate',c)]
   checks={'chunk':c,'outputs':compare(candidate['outputs'],original['outputs']),'states':compare(candidate['state'],original['state'])};checks['passed']=exact(checks['outputs']) and exact(checks['states']);r['cpu_preservation'].append(checks)
   r['gpu_vs_original_cpu'].append({'chunk':c,'outputs':compare(gpu['outputs'],original['outputs']),'states':compare(gpu['state'],original['state'])})
  r['cpu_preservation_passed']=all(c['passed'] for c in r['cpu_preservation'])
  if r['instrumentation_valid'] and r['cpu_preservation_passed']:
   r['stage_checks']=[]
   pre=['value_17','x_65','var_5669','mean_7','var_5672','denom_7']
   for pol in ('CPU','GPU'):
    for c in (0,1):
     actual=got[(pol,'tapped',c)]['outputs'];old=cases[c][pol]['taps'];denom=actual['denom_7'];reference=oracle(denom)
     checks={'policy':pol,'chunk':c,'pre_reciprocal_vs_saved':compare({n:actual[n] for n in pre},{n:old[n] for n in pre}),'seed_vs_saved_reciprocal':compare({'seed':actual[seed]},{'seed':old['var_5675']}),'corrected_vs_rational_oracle':compare({'reciprocal':actual['var_5675']},{'reciprocal':reference}),'stages_vs_original_cpu':compare({n:actual[n] for n in tap_prior['taps']},{n:cases[c]['CPU']['taps'][n] for n in tap_prior['taps']}),'denominator_domain_passed':bool(((denom>=.5)&(denom<=8)).all()),'seed_within_one_neighbor':bool(np.all(np.abs(actual[seed].view(np.uint32).astype(np.int64)-reference.view(np.uint32).astype(np.int64))<=1))};r['stage_checks'].append(checks)
   r['site_gate_passed']=all(exact(c['pre_reciprocal_vs_saved']) and exact(c['seed_vs_saved_reciprocal']) and exact(c['corrected_vs_rational_oracle']) and c['denominator_domain_passed'] and c['seed_within_one_neighbor'] for c in r['stage_checks'])
   warm=next(c for c in r['stage_checks'] if c['policy']=='GPU' and c['chunk']==1);r['warm_normalization_exact']=all(warm['stages_vs_original_cpu'][n]['bitwise_exact'] for n in tap_prior['taps'])
  else:r.update(stage_checks=None,site_gate_passed=False,warm_normalization_exact=False)
  r['prediction_guard']=stats();assert stats()=={'installed':True,'plain_calls':0,'state_calls':12,'gil_releases':12}
  assert all(sha(ROOT/n)==h for n,h in protocol['pins'].items()) and sha(pp)==r['protocol_sha256'] and runtime_provenance()==r['runtime_provenance']
  for key,inv in [('package','variant_files'),('compiled','compiled_files'),('tapped_package','tapped_variant_files'),('tapped_compiled','tapped_compiled_files')]:assert inventory(ROOT/b[key])==b[inv]
  assert inventory(ROOT/base['package'])==base['variant_files'] and inventory(ROOT/base['compiled'])==base['compiled_files']
  r.update(status='completed',passed=r['instrumentation_valid'] and r['cpu_preservation_passed'] and r['site_gate_passed'] and r['warm_normalization_exact'],scope='One reciprocal site on two saved chunks; no full-model qualification or performance claim.')
 except BaseException as e:r.update(status='failed',passed=False,error={'type':type(e).__name__,'message':str(e)});raise
 finally:model=None;gc.collect();update()
 print(json.dumps({k:r[k] for k in ('status','passed','instrumentation_valid','cpu_preservation_passed','site_gate_passed','warm_normalization_exact','prediction_guard')},indent=2))
if __name__=='__main__':main()
