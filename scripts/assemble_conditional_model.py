"""Seal a full conditional-score source/compiled candidate from reviewed group builds."""
from __future__ import annotations
import argparse, hashlib, json, shutil, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def inventory(p):return {str(f.relative_to(p)):sha(f) for f in sorted(p.rglob('*')) if f.is_file()}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--attempt',required=True);p.add_argument('--records-json',required=True);a=p.parse_args()
    if not a.attempt or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in a.attempt):p.error('Invalid attempt')
    out=ROOT/'evidence/05g-integration'/a.attempt;art=ROOT/'artifacts/05g-integration'/a.attempt
    if out.exists() or art.exists():raise FileExistsError('Fresh attempt required')
    out.mkdir(parents=True);art.mkdir(parents=True);source=art/'source';compiled=art/'compiled';source.mkdir();compiled.mkdir()
    r={'status':'starting','arithmetic_policy':'conditional-warm-score','groups':[],'copied':{n:f'artifacts/04d-cpu/full-candidate-6/{n}.mlpackage' for n in ['embedding','head']},'assembler_sha256':sha(__file__)}
    def save():
        t=out/'build.tmp';t.write_text(json.dumps(r,indent=2)+'\n');t.replace(out/'build.json')
    save()
    try:
        from coreml_port.package_identity import verify_candidate_identity,write_candidate_manifest
        from coreml_port.compiled_identity import build_manifest,verify_compiled_identity
        base=ROOT/'artifacts/04d-cpu/full-candidate-6';cache=ROOT/'artifacts/05c-compiled/full-cache-1'
        old=verify_candidate_identity(base,ROOT/'evidence/04d-cpu/full-candidate-6/package-identity.json');assert old['aggregate_sha256']=='5a3e4c7a571e4fc3532377d9b9bb0da023f499c10709216fee7e1001bf4ad2c6'
        verify_compiled_identity(cache,ROOT/'evidence/05c-compiled/full-cache-1/compiled-identity.json',base,old)
        r['source_candidate_identity']=old['aggregate_sha256']
        sources=set(json.loads((ROOT/'evidence/06-gpu/baseline-preservation.json').read_text())['source_runtime_validator_checks'])
        sources.update(['scripts/build_conditional_scores.py','scripts/assemble_conditional_model.py','scripts/qualify_conditional_group.py','portable/model.py','scripts/probe_full_model.py','evidence/03-portable/ACCEPTANCE.json','evidence/01-reference/local-input-validation.json'])
        before={n:sha(ROOT/n) for n in sorted(sources)}
        for n in before:
            dest=out/'source-snapshots'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dest);assert sha(dest)==before[n]
        (out/'source-identity.json').write_text(json.dumps(before,indent=2)+'\n')
        paths=json.loads((ROOT/a.records_json).read_text());assert len(paths)==12
        records={}
        for relative in paths:
            path=ROOT/relative;record=json.loads(path.read_text());g=record['group'];assert g not in records and record['passed'] and record['status']=='completed' and record['graph_check']['passed'] and record['graph_check']['single_model_state_interface_preserved'];assert record['source_candidate_identity']==old['aggregate_sha256'] and record['script_sha256']==before['scripts/build_conditional_scores.py'];assert sha(path.parent/'source-snapshot.py')==record['script_sha256'];records[g]=(relative,record)
        assert set(records)==set(map(str,range(12)))
        for g in map(str,range(12)):
            relative,record=records[g];src=ROOT/record['package'];comp=ROOT/record['compiled']
            assert inventory(src)==record['variant_files'] and inventory(comp)==record['compiled_files'] and inventory(base/f'{g}.mlpackage')==record['source_files']
            shutil.copytree(src,source/f'{g}.mlpackage');shutil.copytree(comp,compiled/f'{g}.mlmodelc')
            assert inventory(source/f'{g}.mlpackage')==record['variant_files'] and inventory(compiled/f'{g}.mlmodelc')==record['compiled_files']
            r['groups'].append({'group':g,'record':relative,'record_sha256':sha(ROOT/relative)});save();print('assembled',g,flush=True)
        copied_lineage=[]
        for n,relative in r['copied'].items():
            src=ROOT/relative;dst=source/f'{n}.mlpackage';shutil.copytree(src,dst);shutil.copytree(cache/f'{n}.mlmodelc',compiled/f'{n}.mlmodelc')
            oldfiles=inventory(src);newfiles=inventory(dst);assert newfiles==oldfiles and inventory(compiled/f'{n}.mlmodelc')==inventory(cache/f'{n}.mlmodelc')
            copied_lineage.append({'component':n,'source_package':relative,'same_file_set':True,'pass':True,'files':[{'path':key,'expected':value,'sha256':newfiles[key],'size_matches':(src/key).stat().st_size==(dst/key).stat().st_size} for key,value in oldfiles.items()]})
        manifest=write_candidate_manifest(source,out/'package-identity.json');r['identity']=manifest['aggregate_sha256']
        cm=build_manifest(compiled,source,manifest);(out/'compiled-identity.json').write_text(json.dumps(cm,indent=2)+'\n')
        verify_compiled_identity(compiled,out/'compiled-identity.json',source,manifest)
        after={n:sha(ROOT/n) for n in before};assert after==before
        assert verify_candidate_identity(base,ROOT/'evidence/04d-cpu/full-candidate-6/package-identity.json')==old
        seal={'pass':True,'sources_unchanged_during_build':True,'before':before,'after':after,'package_identity':manifest['aggregate_sha256'],'copied_lineage':copied_lineage}
        (out/'provenance-verification.json').write_text(json.dumps(seal,indent=2)+'\n')
        r.update(status='completed',passed=True,compiled_identity=cm['aggregate_sha256'],source_root=str(source.relative_to(ROOT)),compiled_root=str(compiled.relative_to(ROOT)))
    except BaseException as e:r.update(status='failed',passed=False,error={'type':type(e).__name__,'message':str(e)});raise
    finally:save()
    print(json.dumps({'passed':True,'source':r['identity'],'compiled':r['compiled_identity']},indent=2))
if __name__=='__main__':main()
