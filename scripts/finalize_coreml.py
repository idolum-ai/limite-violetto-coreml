"""Consolidate Step 4 evidence without upgrading failed qualification gates."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'evidence/04-coreml'
def read(n):return json.loads((OUT/n).read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def main():
    cpu=read('validation-CPU_ONLY.json');gpu=read('validation-CPU_AND_GPU.json');gen=read('generation-cpu.json');boundary=read('boundary.json');comparison=read('generation-comparison.json');long=read('longprefill-isolation.json')
    backends={}
    for name,data in [('CPU_ONLY',cpu),('CPU_AND_GPU',gpu)]:
        scores=[c for c in data['comparisons'] if 'scores' in c['name'] or 'teacher' in c['name']]
        backends[name]={'numerical_pass':data['numerical_pass'],'comparisons':len(data['comparisons']),'comparisons_passed':sum(c['pass'] for c in data['comparisons']),'max_score_error':max(c['max_abs'] for c in scores),'score_comparisons':len(scores),'top1_matches':sum(c['top1_match'] for c in scores),'lifecycle_pass':data['lifecycle_pass'],'lifecycle_checks':len(data['checks'])}
    packages={}
    for n in ['embedding',*map(str,range(12)),'head']:
        p=ROOT/'artifacts/04-coreml/model'/f'{n}.mlpackage';files={str(f.relative_to(p)):{'bytes':f.stat().st_size,'sha256':digest(f)} for f in sorted(p.rglob('*')) if f.is_file()};packages[n]={'path':str(p.relative_to(ROOT)),'bytes':sum(f['bytes'] for f in files.values()),'files':files}
    sources=[*sorted((ROOT/'coreml_port').glob('*.py')),*sorted((ROOT/'scripts').glob('*coreml*.py')),ROOT/'experiments/probe_bf16_rounding.py',ROOT/'experiments/debug_bf16_rounding.py']
    result={'updated_at':datetime.now(timezone.utc).isoformat(),'status':'Complete Core ML conversion/generation demonstrated; numerical fidelity FAIL; reliability qualification partial','gates':{'G4.1':'PASS CPU complete-model compilation and completed generation; full GPU-permitted pipeline also executed','G4.2':'FAIL unchanged numerical limits; original NVIDIA-runtime fidelity remains unverified','G4.3':'PARTIAL: recorded lifecycle and isolated capacity checks pass; an initial native crash remains unexplained','G4.4':'PASS for recorded local invocation, pinned inputs and checksummed packages; clean-machine installation not tested'},'backends':backends,'rounding':read('rounding-probe-v2.json'),'capacity':boundary,'single_call_longprefill':long,'generation':{k:gen[k] for k in ['generated_tokens','complete','finish_reason','stop_token_ids','elapsed_seconds']},'generation_comparison':{k:v for k,v in comparison.items() if k!='reference'},'total_package_bytes':sum(v['bytes'] for v in packages.values()),'package_count':len(packages),'weights':'BF16 values preserved as FP32 constants; no weight quantization','activation_arithmetic':'FP32 with explicit finite-range BF16 rounding; not native BF16','cache_dtype':'FP16','actual_device_placement':'not profiled','timing_quality':'diagnostic only, local runs overlap other work','original_acceptance_sha256':digest(ROOT/'evidence/03-portable/ACCEPTANCE.json'),'thresholds_changed':False,'sources':{str(p.relative_to(ROOT)):digest(p) for p in sources},'packages':packages}
    write(OUT/'summary.json',result)
    manifest=json.loads((ROOT/'manifest.json').read_text());manifest['updated_at']=result['updated_at'];manifest['phase']=result['status'];manifest['portable_model']['complete_coreml_conversion_executed']=True;manifest['coreml_model']={'summary':'evidence/04-coreml/summary.json','report':'evidence/04-coreml/REPORT.md','packages':'artifacts/04-coreml/model','capacity':2048,'prefill_chunk':32,'package_count':14,'numerical_fidelity':'FAIL','original_model_fidelity':'UNVERIFIED','gates':result['gates']};manifest['validation']['gate_status'].update(result['gates']);write(ROOT/'manifest.json',manifest)
    print(json.dumps({k:v for k,v in result.items() if k not in ['packages','sources','rounding','capacity','single_call_longprefill']},indent=2))
if __name__=='__main__':main()
