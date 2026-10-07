"""Record qualified CPU cache behavior without rewriting historical failures."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'evidence/03b-cache'
def read(path):return json.loads(path.read_text())
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def main():
    gates=read(OUT/'cpu-sixteen/results.json');stress=read(OUT/'cpu-partitions-v3.json')
    lifecycle=read(OUT/'warmup-lifecycle.json');generation=read(OUT/'cpu-generation.json')
    assert gates['all_local_checks_passed'] and stress['passed'] and lifecycle['passed'] and generation['passed']
    matched=[c for case in stress['cases'] for c in case['matched_prefix_comparisons']]
    result={'updated_at':datetime.now(timezone.utc).isoformat(),'status':'CPU cache qualified on recorded local environment; Metal remains unqualified',
            'device':'Apple M4 CPU','torch_version':'2.7.0','minimum_query_rows':16,'warmup_below_tokens':16,
            'original_acceptance_sha256':hashlib.sha256((ROOT/'evidence/03-portable/ACCEPTANCE.json').read_bytes()).hexdigest(),
            'numerical_comparisons':len(gates['comparisons']),'numerical_passed':sum(c['passed'] for c in gates['comparisons']),
            'numerical_max_abs':max(c['max_abs'] for c in gates['comparisons']),
            'behavioral_and_summary_checks':len(gates['checks']),'all_gate_checks_passed':gates['all_local_checks_passed'],
            'extra_partition_cases':len(stress['cases']),'extra_matched_prefix_comparisons':len(matched),
            'extra_hidden_max_abs':max(c['hidden']['max_abs'] for c in matched),'extra_score_max_abs':max(c['scores']['max_abs'] for c in matched),
            'warmup_lifecycle_checks':len(lifecycle['checks']),'warmup_lifecycle_passed':lifecycle['passed'],
            'generation':{'generated_tokens':generation['generation']['generated_tokens'],'complete':generation['generation']['complete'],'finish_reason':generation['generation']['finish_reason'],'stop_token_ids':generation['generation']['stop_token_ids'],'boxed_391_after_think':generation['boxed_391_after_think'],'audits':len(generation['audits']),'audit_max_score_abs':max(x['scores']['max_abs'] for x in generation['audits'])},
            'limitations':['Metal cache still fails original thresholds','Original NVIDIA runtime not executed','Published actor EOS conflict retained; explicit stop option used','Qualification is empirical for recorded inputs, Mac and Torch version'],
            'sources':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['portable/model.py','portable/generate.py','scripts/validate_portable.py','scripts/stress_cpu_cache.py','scripts/audit_cpu_generation.py']}}
    write(OUT/'summary.json',result)
    manifest=read(ROOT/'manifest.json');manifest['updated_at']=result['updated_at'];manifest['phase']='Portable CPU cache qualified; complete Core ML conversion not started; original-runtime fidelity unverified'
    manifest['validation']['gate_status']['G3.3']='PASS on qualified CPU path; FAIL on original and padded Metal paths'
    manifest['portable_model'].update(summary='evidence/03b-cache/summary.json',report='evidence/03b-cache/REPORT.md',cached_numerical_comparison='PASS CPU sixteen-row path; FAIL MPS',attention_minimum_query_rows_cpu=16,cache_warmup_below_tokens_cpu=16)
    manifest['cache_followup']={'summary':'evidence/03b-cache/summary.json','original_failures':'evidence/03-portable/summary.json','thresholds_changed':False,'cpu_passed':True,'mps_passed':False}
    write(ROOT/'manifest.json',manifest);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
