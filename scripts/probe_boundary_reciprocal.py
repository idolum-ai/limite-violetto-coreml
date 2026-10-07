"""Native external-seed reciprocal qualification and separate baseline reproduction."""
from pathlib import Path
import argparse,gc,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.probe_reciprocal_refinement import sha,read,save,inventory,compare,prepare_cases,array_sha,SOURCE
HELPER=ROOT/'scripts/boundary_reciprocal.py'
OLD=ROOT/'scripts/probe_reciprocal_refinement.py'

def main():
 p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);p.add_argument('--protocol',required=True);a=p.parse_args()
 pp=ROOT/a.protocol;protocol=read(pp)
 assert a.attempt==protocol['attempt'] and a.attempt and all(c in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in a.attempt)
 assert protocol['expected_predictions']==36
 for path,h in protocol['pins'].items():assert sha(ROOT/path)==h,(path,'pin mismatch')
 assert str(Path(__file__).relative_to(ROOT)) in protocol['pins'] and str(HELPER.relative_to(ROOT)) in protocol['pins']
 source=read(SOURCE);assert source['passed'] and source['instrumentation_valid']
 out=ROOT/'evidence/06f-gpu-reciprocal'/a.attempt;art=ROOT/'artifacts/06f-gpu-reciprocal'/a.attempt
 assert not out.exists() and not art.exists();out.mkdir();art.mkdir(parents=True)
 for name,path in [('source-snapshot.py',Path(__file__)),('helper-snapshot.py',HELPER),('protocol.json',pp),('case-builder-snapshot.py',OLD)]: (out/name).write_bytes(path.read_bytes())
 result={'status':'starting','passed':False,'protocol_sha256':sha(pp),'pins':protocol['pins'],'models':{},'cases':[],'calls':[],'actual_device_placement':'unverified; compute policy only'}
 def update():save(out/'results.json',result)
 update();model=None
 try:
  import numpy as np
  import torch
  import coremltools as ct
  from scripts.boundary_reciprocal import correct_reciprocal, mil_correct_reciprocal
  from coreml_port.prediction_guard import install,stats
  from scripts.runtime_provenance import runtime_provenance
  from collections import Counter
  torch.set_num_threads(4)
  result['runtime_provenance']=runtime_provenance();assert result['runtime_provenance']==source['runtime_provenance']
  cases=prepare_cases(source,protocol['source_archives']);seedcases=[]
  for c in cases:
   for arm in ('nearest','lower','upper'):
    seed=c['reference'].copy() if arm=='nearest' else np.nextafter(c['reference'],np.float32(-np.inf if arm=='lower' else np.inf))
    seedcases.append({'name':c['name']+'/'+arm,'d':c['d'],'seed':seed,'reference':c['reference']})
   if 'GPU' in c['saved']:seedcases.append({'name':c['name']+'/captured-gpu','d':c['d'],'seed':c['saved']['GPU'],'reference':c['reference']})
  assert len(seedcases)==14
  assert {c['name']:{'d':array_sha(c['d']),'seed':array_sha(c['seed'])} for c in seedcases}==protocol['seed_case_hashes']
  def archive(label,arrays):
   f=out/(label+'.npz');f.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(f,**arrays)
   return {'path':str(f.relative_to(ROOT)),'sha256':sha(f),'shapes':{n:list(x.shape) for n,x in arrays.items()},'dtypes':{n:str(x.dtype) for n,x in arrays.items()}}
  host=[]
  for c in seedcases:
   with torch.inference_mode():value=correct_reciprocal(torch.from_numpy(c['d'].copy()),torch.from_numpy(c['seed'].copy())).numpy()
   check=compare(value,c['reference']);assert check['bitwise_exact'],(c['name'],check)
   meta=archive('cases/'+c['name'],{n:c[n] for n in ('d','seed','reference')});result['cases'].append({'name':c['name'],'archive':meta});host.append({'name':c['name'],**check})
  result['host_checks']=host;update()
  install();assert stats()=={'installed':True,'plain_calls':0,'state_calls':0,'gil_releases':0}
  from coremltools.converters.mil import Builder as mb
  from coremltools.converters.mil.mil import get_new_symbol
  symbol=get_new_symbol()
  @mb.program(input_specs=[mb.TensorSpec(shape=(1,symbol,1))],opset_version=ct.target.macOS15)
  def control(d):return mb.real_div(x=mb.fill_like(ref_tensor=d,value=np.float32(1)),y=d,name='value')
  @mb.program(input_specs=[mb.TensorSpec(shape=(1,symbol,1)),mb.TensorSpec(shape=(1,symbol,1))],opset_version=ct.target.macOS15)
  def correction(d,seed):return mb.identity(x=mil_correct_reciprocal(mb,d,seed),name='value')
  for arm,program in [('control',control),('correction',correction)]:
   shape=(1,ct.RangeDim(1,32,default=12),1)
   cm=ct.convert(program,source='milinternal',inputs=[ct.TensorType(name=n,shape=shape,dtype=np.float32) for n in (('d',) if arm=='control' else ('d','seed'))],convert_to='mlprogram',compute_precision=ct.precision.FLOAT32,minimum_deployment_target=ct.target.macOS15,pass_pipeline=ct.PassPipeline.EMPTY,skip_model_load=True)
   spec=cm.get_spec();ops=spec.mlProgram.functions['main'].block_specializations['CoreML8'].operations;counts=dict(Counter(op.type for op in ops))
   assert all(not op.blocks for op in ops)
   assert counts.get('real_div',0)==(1 if arm=='control' else 0) and not counts.get('inverse',0)
   package=art/(arm+'.mlpackage');cm.save(str(package));compiled=art/(arm+'.mlmodelc')
   returned=ct.models.utils.compile_model(str(package),destination_path=str(compiled));assert Path(returned).resolve()==compiled.resolve()
   result['models'][arm]={'package':str(package.relative_to(ROOT)),'compiled':str(compiled.relative_to(ROOT)),'package_files':inventory(package),'compiled_files':inventory(compiled),'operation_counts':counts,'operations_sha256':hashlib.sha256(b''.join(op.SerializeToString(deterministic=True) for op in ops)).hexdigest()};update()
   del cm
  native={}
  for policy in ('CPU','GPU'):
   for arm in ('control','correction'):
    model=ct.models.CompiledMLModel(str(ROOT/result['models'][arm]['compiled']),compute_units=ct.ComputeUnit.CPU_ONLY if policy=='CPU' else ct.ComputeUnit.CPU_AND_GPU)
    for c in cases if arm=='control' else seedcases:
     payload={'d':c['d'].copy()}
     if arm=='correction':payload['seed']=c['seed'].copy()
     got={n:x.copy() for n,x in model.predict(payload).items()};assert set(got)=={'value'}
     assert all(compare(payload[n],c[n])['bitwise_exact'] for n in payload)
     check=compare(got['value'],c['reference']);native[(policy,arm,c['name'])]=got['value'];result['calls'].append({'policy':policy,'arm':arm,'case':c['name'],'archive':archive('native/'+policy+'/'+arm+'/'+c['name'],got),'vs_oracle':check,'input_preserved':True});update()
    del model;model=None;gc.collect()
  reproduction=[];preservation=[]
  for c in cases:
   for policy,want in c['saved'].items():reproduction.append({'case':c['name'],'policy':policy,**compare(native[(policy,'control',c['name'])],want)})
   preservation.append({'case':c['name'],**compare(native[('CPU','correction',c['name']+'/nearest')],native[('CPU','control',c['name'])])})
  correction_pass=all(c['vs_oracle']['bitwise_exact'] for c in result['calls'] if c['arm']=='correction')
  control_pass=all(c['vs_oracle']['bitwise_exact'] for c in result['calls'] if c['arm']=='control' and c['policy']=='CPU')
  preserve_pass=all(x['bitwise_exact'] for x in preservation)
  repro_pass=all(x['bitwise_exact'] for x in reproduction)
  result.update(prediction_guard=stats(),reproduction_checks=reproduction,cpu_preservation_checks=preservation,gates={'C3_saved_baseline_reproduced':repro_pass,'C3b_external_seed_arithmetic':correction_pass,'cpu_control_oracle':control_pass,'cpu_preservation':preserve_pass},seeded_component_passed=correction_pass and control_pass and preserve_pass,baseline_reproduced=repro_pass)
  assert stats()=={'installed':True,'plain_calls':36,'state_calls':0,'gil_releases':36}
  assert sha(pp)==result['protocol_sha256'] and all(sha(ROOT/k)==h for k,h in protocol['pins'].items()) and runtime_provenance()==result['runtime_provenance']
  for rec in result['models'].values():assert inventory(ROOT/rec['package'])==rec['package_files'] and inventory(ROOT/rec['compiled'])==rec['compiled_files']
  result.update(status='completed',passed=result['seeded_component_passed'] and repro_pass,decision='component passes; group qualification required' if correction_pass and repro_pass else 'separate in-model diagnostic required; no repair claim' if correction_pass else 'native correction failed')
 except BaseException as e:result.update(status='failed',passed=False,error={'type':type(e).__name__,'message':str(e)});raise
 finally:model=None;gc.collect();update()
 print(json.dumps({k:result[k] for k in ('status','passed','gates','prediction_guard','decision')},indent=2))
if __name__=='__main__':main()
