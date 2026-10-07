"""Control an output-only tap clone of the signed-even group0 candidate."""
from pathlib import Path
import argparse,gc,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
BASE=ROOT/'evidence/06d-gpu-rounding/group0-1/results.json'
PRIOR=ROOT/'evidence/06d-gpu-rounding/group0-replay-1/results.json'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def inventory(p):return {str(x.relative_to(p)):sha(x) for x in sorted(p.rglob('*')) if x.is_file()}
def save(p,r):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(r,indent=2,sort_keys=True,allow_nan=False)+'\n');t.replace(p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--build',required=True);p.add_argument('--attempt',required=True);a=p.parse_args()
 assert a.attempt and all(c in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in a.attempt)
 out=ROOT/'evidence/06d-gpu-rounding'/a.attempt;out.mkdir(parents=True,exist_ok=False)
 bp=ROOT/a.build;b=read(bp);base=read(BASE);prior=read(PRIOR)
 assert b['status']=='completed' and b['passed'] and b['graph_check']['passed']
 assert base['passed'] and prior['status']=='completed' and prior['cpu_preservation_passed']
 pins={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),bp,BASE,PRIOR)}
 (out/'source-snapshot.py').write_bytes(Path(__file__).read_bytes());(out/'build-snapshot.json').write_bytes(bp.read_bytes())
 r={'status':'starting','pins':pins,'files':{},'calls':[],'controls':[],'actual_device_placement':'unverified; compute policy only'}
 def update():save(out/'results.json',r)
 update();model=None
 try:
  import coremltools as ct
  import numpy as np
  from coreml_port.prediction_guard import install,stats
  from scripts.runtime_provenance import runtime_provenance
  from scripts.probe_gpu_group0 import compare_array
  assert inventory(ROOT/base['package'])==base['variant_files'] and inventory(ROOT/base['compiled'])==base['compiled_files']
  assert inventory(ROOT/b['package'])==b['variant_files'] and inventory(ROOT/b['compiled'])==b['compiled_files']
  assert b['source_build_sha256']==sha(BASE) and b['source_files']==base['variant_files']
  taps=[x['name'] for x in b['graph_check']['added_taps']];assert taps and len(taps)==len(set(taps))
  r['taps']=taps;r['runtime_provenance']=runtime_provenance();assert r['runtime_provenance']==prior['runtime_provenance']
  names=[f'inner_{k}_cache_{i}' for i in range(4) for k in ('k','v')]
  def load(label):
   info=prior['files'][label];path=ROOT/info['path'];assert sha(path)==info['sha256']
   with np.load(path,allow_pickle=False) as z:return {n:z[n].copy() for n in z.files}
  def archive(label,arrays):
   path=out/(label+'.npz');path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**arrays)
   info={'path':str(path.relative_to(ROOT)),'sha256':sha(path),'shapes':{n:list(v.shape) for n,v in arrays.items()},'dtypes':{n:str(v.dtype) for n,v in arrays.items()}};r['files'][label]=info;return info
  def compare(actual,expected):
   assert set(actual)==set(expected)
   return {n:{**compare_array(expected[n],actual[n]),'bitwise_exact':actual[n].shape==expected[n].shape and actual[n].dtype==expected[n].dtype and actual[n].tobytes()==expected[n].tobytes()} for n in actual}
  def exact(c):return all(v['bitwise_exact'] for v in c.values())
  cases={c:{'inputs':load(f'chunk{c}/inputs'),'before':load(f'chunk{c}/before'),**{pol:{'outputs':load(f'chunk{c}/candidate-{pol}/outputs'),'state':load(f'chunk{c}/candidate-{pol}/state-after')} for pol in ('CPU','GPU')}} for c in (0,1)}
  install();assert stats()=={'installed':True,'plain_calls':0,'state_calls':0,'gil_releases':0}
  got={}
  for pol in ('CPU','GPU'):
   for arm,compiled in [('original',base['compiled']),('tapped',b['compiled'])]:
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
    original=got[(pol,'original',c)];tapped=got[(pol,'tapped',c)]
    control={'policy':pol,'chunk':c,'original_outputs_vs_saved':compare(original['outputs'],cases[c][pol]['outputs']),'original_states_vs_saved':compare(original['state'],cases[c][pol]['state']),'tapped_endpoints_vs_original':compare({n:tapped['outputs'][n] for n in ('y','tap23_out')},original['outputs']),'tapped_states_vs_original':compare(tapped['state'],original['state'])}
    control['passed']=all(exact(control[n]) for n in ('original_outputs_vs_saved','original_states_vs_saved','tapped_endpoints_vs_original','tapped_states_vs_original'));r['controls'].append(control)
  r['instrumentation_valid']=all(x['passed'] for x in r['controls'])
  r['tap_comparisons']=[{'chunk':c,'comparisons':compare({n:got[('GPU','tapped',c)]['outputs'][n] for n in taps},{n:got[('CPU','tapped',c)]['outputs'][n] for n in taps})} for c in (0,1)] if r['instrumentation_valid'] else None
  r['prediction_guard']=stats();assert stats()=={'installed':True,'plain_calls':0,'state_calls':8,'gil_releases':8}
  assert all(sha(ROOT/p)==h for p,h in pins.items()) and runtime_provenance()==r['runtime_provenance']
  for rec in (base,b):assert inventory(ROOT/rec['package'])==rec['variant_files'] and inventory(ROOT/rec['compiled'])==rec['compiled_files']
  r.update(status='completed',passed=r['instrumentation_valid'],scope='Same saved inputs/cache; diagnostic only, no full-model qualification.')
 except BaseException as e:r.update(status='failed',passed=False,error={'type':type(e).__name__,'message':str(e)});raise
 finally:model=None;gc.collect();update()
 print(json.dumps({'status':r['status'],'instrumentation_valid':r['instrumentation_valid'],'guard':r['prediction_guard']},indent=2))
if __name__=='__main__':main()
