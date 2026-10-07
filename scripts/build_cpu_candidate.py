"""Build the current experimental CPU candidate in an isolated package root."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from coreml_port.package_identity import write_candidate_manifest
p=argparse.ArgumentParser();p.add_argument('--attempt',default='full-candidate-1');p.add_argument('--enhanced',action='store_true');p.add_argument('--source-exp',action='store_true');p.add_argument('--dynamic-product',action='store_true');p.add_argument('--paired-product',action='store_true');p.add_argument('--tiny-sequential',action='store_true');p.add_argument('--tiny-native',action='store_true');args=p.parse_args()
if args.tiny_native and args.tiny_sequential:p.error('--tiny-native and --tiny-sequential are mutually exclusive')
if args.tiny_native and not args.paired_product:p.error('--tiny-native requires --paired-product')
if (args.paired_product or args.tiny_sequential) and not args.dynamic_product:p.error('--paired-product and --tiny-sequential require --dynamic-product')
if args.dynamic_product and not args.source_exp:p.error('--dynamic-product requires --source-exp')
if args.source_exp and not args.enhanced:p.error('--source-exp requires --enhanced')
if not args.attempt.replace('-','').isalnum():raise ValueError('Use a simple attempt name.')
out=ROOT/'evidence/04d-cpu'/args.attempt;art=ROOT/'artifacts/04d-cpu'/args.attempt
out.mkdir(parents=True,exist_ok=False);art.mkdir(parents=True,exist_ok=False)
source_names=['coreml_port/graphs.py','coreml_port/ordered_linear.py','coreml_port/ordered_attention.py','scripts/convert_coreml.py','scripts/build_cpu_candidate.py','coreml_port/package_identity.py']
if args.enhanced:source_names+=['coreml_port/ordered_norm.py','coreml_port/ordered_xsa.py','coreml_port/bf16_gelu.py']
if args.source_exp:source_names+=['coreml_port/source_exp.py']
if args.dynamic_product:source_names+=['coreml_port/dynamic_queries.py']
if args.paired_product:source_names+=['coreml_port/paired_product.py']
if args.tiny_sequential:source_names+=['coreml_port/tiny_scores.py']
if args.tiny_native:source_names+=['coreml_port/tiny_native_scores.py']
sources={}
for n in source_names:
    data=(ROOT/n).read_bytes();sources[n]=hashlib.sha256(data).hexdigest()
    snapshot=out/'source-snapshots'/n;snapshot.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes(data)
(out/'source-identity.json').write_text(json.dumps(sources,indent=2)+'\n')
copies={'head':'artifacts/04d-cpu/full-candidate-1/head.mlpackage'} if args.enhanced else {'embedding':'artifacts/04-coreml/model/embedding.mlpackage','0':'artifacts/04d-cpu/ordered-softmax/0.mlpackage'}
if args.source_exp:copies={'head':'artifacts/04d-cpu/full-candidate-2/head.mlpackage','embedding':'artifacts/04d-cpu/full-candidate-2/embedding.mlpackage'}
for name,source in copies.items():
    shutil.copytree(ROOT/source,art/(name+'.mlpackage'))
report={'copied':copies,'conversions':[],'status':'building','enhanced':args.enhanced,'source_exp':args.source_exp,'dynamic_product':args.dynamic_product,'paired_product':args.paired_product,'tiny_sequential':args.tiny_sequential,'tiny_native':args.tiny_native,'thresholds_changed':False}
def save():(out/'build.json').write_text(json.dumps(report,indent=2)+'\n')
def convert(name):
    command=[sys.executable,str(ROOT/'scripts/convert_coreml.py'),name,'--arithmetic','ordered','--softmax','ordered','--output-dir',str(art),'--evidence-dir',str(out)]
    if args.enhanced:command+=['--norm','ordered','--xsa','ordered','--gelu','lookup']
    if args.source_exp:command+=['--exp','source']
    if args.dynamic_product:command+=['--queries','dynamic-product']
    if args.paired_product:command+=['--product','paired']
    if args.tiny_sequential:command+=['--score-dot','tiny-sequential']
    if args.tiny_native:command+=['--score-dot','tiny-native']
    start=time.time()
    with (out/(name+'.log')).open('w') as f:r=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    return {'component':name,'command':command,'exit_code':r.returncode,'seconds':time.time()-start}
save()
with ThreadPoolExecutor(max_workers=2) as pool:
    tasks=[pool.submit(convert,n) for n in ['embedding',*map(str,range(12)),'head'] if n not in copies]
    for future in as_completed(tasks):
        result=future.result();report['conversions'].append(result);save();print(result,flush=True)
if any(x['exit_code'] for x in report['conversions']):
    report['status']='conversion failure';save();raise SystemExit(1)
report['identity']=write_candidate_manifest(art,out/'package-identity.json')['aggregate_sha256'];report['status']='converted; runtime validation pending';save();print(report['status'],report['identity'],flush=True)
