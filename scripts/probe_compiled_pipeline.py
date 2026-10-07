"""Fresh-process startup and first-prediction probe of the explicit full loader."""
from __future__ import annotations
import time
ENTRY = time.perf_counter()
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',required=True);p.add_argument('--identity-manifest',required=True)
    p.add_argument('--compiled-dir');p.add_argument('--compiled-manifest')
    p.add_argument('--evidence-dir',required=True);args=p.parse_args()
    args.compute='CPU_ONLY';args.tiny_prefix_replay=True
    out=(ROOT/args.evidence_dir).resolve()
    if not out.is_relative_to(ROOT/'evidence/05c-compiled'):p.error('Output must be under evidence/05c-compiled')
    out.mkdir(parents=True,exist_ok=False)
    result={'status':'starting','script_sha256':sha(__file__),'host':{'platform':platform.platform(),'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'load_average':os.getloadavg()},'cache_state':'uncontrolled OS caches; persistent compiled path reused'}
    (out/'source-snapshot.py').write_bytes(Path(__file__).read_bytes())
    def save():
        temp=out/'results.tmp';temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(out/'results.json')
    save()
    try:
        import numpy as np
        import torch
        torch.set_num_threads(4)
        from coreml_port.loader_selection import create_pipeline,loader_provenance,runtime_adapter_name
        from scripts.runtime_provenance import runtime_provenance
        result['runtime_adapter']=runtime_adapter_name(args);result['loader_provenance']=loader_provenance(args)
        result['runtime_provenance']=runtime_provenance()
        result['entry_to_imports_seconds']=time.perf_counter()-ENTRY
        start=time.perf_counter();model=create_pipeline(args);result['initialization_seconds']=time.perf_counter()-start
        result['component_load_seconds']=model.load_seconds
        result['identity_verification_seconds']=getattr(model,'identity_verification_seconds',None)
        result['source_spec_seconds']=getattr(model,'source_spec_seconds',None)
        result['entry_to_ready_seconds']=time.perf_counter()-ENTRY
        result['package_identity']=model.package_identity
        result['status']='loaded';save()
        fixture=ROOT/'evidence/04-coreml/generation-comparison.json'
        ref=json.loads(fixture.read_text())['reference'];stream=ref['prompt_ids']+ref['generated_ids']
        scored=ROOT/'evidence/05-practical/coreml-pilot-1/scores.npz'
        pilot=json.loads((scored.parent/'results.json').read_text())
        if sha(scored)!=pilot['scores_sha256'] or pilot['candidate_identity']!=model.package_identity['aggregate_sha256']:
            raise ValueError('Source score fixture identity mismatch')
        result['input_fixture_sha256']=sha(fixture);result['score_fixture_sha256']=sha(scored)
        result['prompt_ids']=stream[:32];result['forced_next_token']=stream[32]
        start=time.perf_counter();req=model.request();result['request_create_seconds']=time.perf_counter()-start
        with np.load(scored) as expected:
            try:
                start=time.perf_counter();first=model.forward(stream[:32],req);result['prefill32_seconds']=time.perf_counter()-start
                result['entry_to_first_scores_seconds']=time.perf_counter()-ENTRY
                result['first_scores_exact']=bool(np.array_equal(first,expected['p32_s0']))
                start=time.perf_counter();second=model.forward([stream[32]],req);result['decode_seconds']=time.perf_counter()-start
                result['decode_scores_exact']=bool(np.array_equal(second,expected['p32_s1']))
            finally:
                req.close();result['request_disposed']=req.closed and not req.states
        from coreml_port.prediction_guard import stats
        result['prediction_guard']=stats()
        result['passed']=result['first_scores_exact'] and result['decode_scores_exact'] and result['request_disposed']
        result['status']='completed'
        del model;gc.collect()
    except BaseException as exc:
        result.update(status='failed',passed=False,failure={'type':type(exc).__name__,'message':str(exc)});raise
    finally:
        result['entry_to_finish_seconds']=time.perf_counter()-ENTRY
        result['peak_process_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        save()
    print(json.dumps({k:v for k,v in result.items() if k.endswith('_seconds') or k in ['status','passed','first_scores_exact','decode_scores_exact']},indent=2),flush=True)
    return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
