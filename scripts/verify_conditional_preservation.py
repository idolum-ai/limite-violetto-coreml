"""Reverify both sealed CPU cohorts and pinned sources after integration experiments."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);a=parser.parse_args();out=ROOT/a.output
    if out.exists():raise FileExistsError(out)
    from coreml_port.compiled_identity import verify_compiled_identity
    from scripts.summarize_conditional_milestone import qualified_context
    from scripts.runtime_provenance import runtime_provenance
    r={'passed':False,'script_sha256':sha(__file__),'cohorts':{},'sources':{}}
    try:
        context=qualified_context();assert not context['errors'],context['errors']
        r['qualified_sources']=context['expected_sources'];r['runtime_provenance']=runtime_provenance();assert r['runtime_provenance']==context['milestone']['current_runtime_provenance']
        paths={
          'baseline':('artifacts/04d-cpu/full-candidate-6','evidence/04d-cpu/full-candidate-6/package-identity.json','artifacts/05c-compiled/full-cache-1','evidence/05c-compiled/full-cache-1/compiled-identity.json','5a3e4c7a571e4fc3532377d9b9bb0da023f499c10709216fee7e1001bf4ad2c6','a838dd36e8f87c6b9fc0dd7865d0912651c6ab3399fa299d10ff12d788bcb031'),
          'conditional':('artifacts/05g-integration/full-1/source','evidence/05g-integration/full-1/package-identity.json','artifacts/05g-integration/full-1/compiled','evidence/05g-integration/full-1/compiled-identity.json','c7cc54a308bb7c379f6d55c777a62666e86921f27245115dc185597efc65a46b','149d74d2e339fabdafeec8cf5ad6766fdd2e2754c15f5256c7dc69d91b932557')}
        for name,(src,sm,cache,cm,sid,cid) in paths.items():
            result=verify_compiled_identity(ROOT/cache,ROOT/cm,ROOT/src,ROOT/sm)
            assert result['source_candidate_identity']==sid and result['aggregate_sha256']==cid
            r['cohorts'][name]={'source':sid,'compiled':cid,'passed':True};print(name,'source/cache verified',flush=True)
        sources=json.loads((ROOT/'evidence/05g-integration/full-1/source-identity.json').read_text())
        for name,expected in sources.items():
            actual=sha(ROOT/name);r['sources'][name]={'expected':expected,'actual':actual,'passed':actual==expected}
        assert all(x['passed'] for x in r['sources'].values())
        assert qualified_context()['expected_sources']==context['expected_sources']
        r['passed']=True
    except BaseException as e:r['error']={'type':type(e).__name__,'message':str(e)};raise
    finally:out.write_text(json.dumps(r,indent=2)+'\n')
    print('preservation PASS',flush=True)
if __name__=='__main__':main()
