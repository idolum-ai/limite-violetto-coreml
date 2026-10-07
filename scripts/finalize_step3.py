"""Collect preserved Step 3 results into the manifest and evidence summary."""
from collections import Counter
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/03-portable'

def read(path):return json.loads(path.read_text())
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def stats(run):
    r=read(OUT/run/'results.json')
    full=[x for x in r['comparisons'] if '/full/' in x['name']]
    cached=[x for x in r['comparisons'] if '/full/' not in x['name']]
    scores=[x for x in cached if x['kind']=='scores'];hidden=[x for x in cached if x['kind']=='hidden']
    return {'path':f'{run}/results.json','device':r['device'],'checks':len(r['checks']),
            'checks_passed':sum(x['passed'] for x in r['checks']),
            'full_prefix_comparisons':len(full),'full_prefix_passed':sum(x['passed'] for x in full),
            'full_prefix_max_abs':max(x['max_abs'] for x in full),
            'cache_comparisons':len(cached),'cache_passed':sum(x['passed'] for x in cached),
            'cache_max_score_abs':max(x['max_abs'] for x in scores),
            'cache_max_hidden_relative_rmse':max(x['relative_rmse'] for x in hidden),
            'cache_min_hidden_cosine':min(x['cosine'] for x in hidden),
            'cache_top1_agreement':sum(x['top1_match'] for x in scores),'cache_score_comparisons':len(scores),
            'peak_process_rss_gib':r['peak_process_rss_gib'],
            'all_local_checks_passed':r['all_local_checks_passed']}

def main():
    weight=read(OUT/'run3/weight-mapping.json')
    samples=[]
    for run in ['run2','run3']:
        for n,g in enumerate(read(OUT/run/'results.json')['generation']):
            samples.append({'path':f'{run}/results.json','case_index':n,'prompt':g['prompt'],'generated_tokens':g['generated_tokens'],'finish_reason':g['finish_reason'],'complete':g['complete'],'greedy':g['greedy']})
    for name in ['cli-math','cli-math-sampled','cli-math-full-budget','cli-math-explicit-stop']:
        g=read(OUT/f'{name}.json')
        final=g['text'].rsplit('</think>',1)[-1]
        samples.append({'path':f'{name}.json','prompt':'Compute 17 * 23.','generated_tokens':g['generated_tokens'],'finish_reason':g['finish_reason'],'complete':g['complete'],'greedy':g['greedy'],'seed':g['seed'],'boxed_391_after_think':'</think>' in g['text'] and '\\boxed{391}' in final,'completion_policy':g.get('completion_policy','published_eos_only'),'stop_token_ids':g.get('stop_token_ids',[g['eos_token_id']])})
    summary={'phase':'Step 3 implemented; validation partial','updated_at':datetime.now(timezone.utc).isoformat(),
             'gates':{'G3.1':'PASS','G3.2':'BLOCKED for independent original runtime; local source comparison PASS','G3.3':'FAIL on CPU and Metal under unchanged criteria','G3.4':'PARTIAL: input/reset/stop mechanics pass; published versus emitted EOS discrepancy unresolved; explicit extra-stop option verified'},
             'checkpoint_sha256':weight['checkpoint_sha256'],'tensor_count':weight['tensor_count'],
             'parameter_count':weight['parameter_count'],'tensor_dtype_counts':dict(Counter(x['dtype'] for x in weight['inventory'])),
             'acceptance_sha256':hashlib.sha256((OUT/'ACCEPTANCE.json').read_bytes()).hexdigest(),
             'cpu':stats('run3'),'mps':stats('run2'),'head_and_stop_checks':read(OUT/'head-and-stops.json'),
             'generation':samples,'eos_observation':'eos-observation.json','complete_coreml_conversion_executed':False,'original_cuda_runtime_executed':False,
             'next_step':'Investigate and qualify cached numerical variation before full-model conversion; original CUDA output captures remain needed for fidelity.'}
    write(OUT/'summary.json',summary)
    manifest=read(ROOT/'manifest.json')
    manifest['updated_at']=summary['updated_at']
    manifest['phase']=summary['phase']+'; cache numerical gate FAIL; original-model fidelity unverified'
    manifest['model']['checkpoint'].update(downloaded=True,locally_verified=True,download_evidence='evidence/03-portable/checkpoint-download.json')
    manifest['validation'].update(model_inference_executed=True,numerical_tolerances='evidence/03-portable/ACCEPTANCE.json: frozen local comparison limits; original-runtime tolerances remain unqualified')
    manifest['validation']['gate_status'].update(summary['gates'])
    manifest['portable_model']={'summary':'evidence/03-portable/summary.json','report':'evidence/03-portable/REPORT.md','tensor_count':weight['tensor_count'],'parameter_count':weight['parameter_count'],'real_weights_used':True,'full_prefix_local_comparison':'PASS CPU and MPS','cached_numerical_comparison':'FAIL CPU and MPS','complete_coreml_conversion_executed':False,'original_model_fidelity':'UNVERIFIED'}
    checkpoint={'path':'sources/model/model.safetensors','size':2070811520,'sha256':weight['checkpoint_sha256']}
    manifest['source_files']=[x for x in manifest['source_files'] if x['path']!=checkpoint['path']]+[checkpoint]
    write(ROOT/'manifest.json',manifest)
    print(json.dumps({'gates':summary['gates'],'generation':samples},indent=2))

if __name__=='__main__':main()
