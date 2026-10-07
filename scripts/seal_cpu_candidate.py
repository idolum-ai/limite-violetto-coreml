"""Verify source stability and copied-package lineage for full CPU trial 1."""
from pathlib import Path
import argparse,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from coreml_port.package_identity import verify_candidate_identity
p=argparse.ArgumentParser();p.add_argument('--attempt',default='full-candidate-1');args=p.parse_args()
if not args.attempt.replace('-','').isalnum():raise ValueError('Use a simple attempt name.')
out=ROOT/'evidence/04d-cpu'/args.attempt;art=ROOT/'artifacts/04d-cpu'/args.attempt
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
before=json.loads((out/'source-identity.json').read_text());after={n:digest(ROOT/n) for n in before}
identity=verify_candidate_identity(art,out/'package-identity.json')
copied=json.loads((out/'build.json').read_text())['copied'];lineage=[]
for name,source in copied.items():
    package=next(p for p in identity['packages'] if p['name']==name);parent=Path(name+'.mlpackage')
    checks=[{'source':str(Path(source)/Path(v['path']).relative_to(parent)),'size_matches':(ROOT/source/Path(v['path']).relative_to(parent)).stat().st_size==v['size'],'sha256':digest(ROOT/source/Path(v['path']).relative_to(parent)),'expected':v['sha256']} for v in package['files']]
    source_paths=sorted(str(p.relative_to(ROOT/source)) for p in (ROOT/source).rglob('*') if p.is_file())
    destination_paths=sorted(str(Path(v['path']).relative_to(parent)) for v in package['files'])
    lineage.append({'component':name,'source_package':source,'files':checks,'same_file_set':source_paths==destination_paths,'pass':source_paths==destination_paths and all(c['size_matches'] and c['sha256']==c['expected'] for c in checks)})
r={'sources_unchanged_during_build':before==after,'before':before,'after':after,'copied_lineage':lineage,'package_identity':identity['aggregate_sha256'],'handoff_script_hashes':{n:digest(ROOT/n) for n in ['scripts/build_cpu_candidate.py','scripts/seal_cpu_candidate.py','coreml_port/package_identity.py','coreml_port/runtime.py','scripts/validate_coreml.py']}}
r['pass']=r['sources_unchanged_during_build'] and all(x['pass'] for x in lineage)
(out/'provenance-verification.json').write_text(json.dumps(r,indent=2)+'\n');print('provenance verified',r['pass'],flush=True);raise SystemExit(not r['pass'])
