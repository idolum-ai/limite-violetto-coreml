"""Sequential bounded suite: separate processes keep memory/timing attributable."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
cases=[("attention-small-local-attempt6",["attention","--attempt","6"]),
       ("attention-small-global-attempt2",["attention","--global-attention","--attempt","2"]),
       ("mixer-small-attempt2",["mixer","--attempt","2"]),
       ("attention-production-local-attempt2",["attention","--production","--attempt","2"]),
       ("attention-production-global-attempt2",["attention","--production","--global-attention","--attempt","2"]),
       ("mixer-production-attempt2",["mixer","--production","--attempt","2"])]
results=[]
for name,args in cases:
    log=ROOT/'evidence/02-components'/f'{name}.log'
    cmd=[sys.executable,str(ROOT/'experiments/run_components.py'),*args]
    print('Starting',name,flush=True)
    start=time.perf_counter()
    with log.open('w') as output:
        try:
            p=subprocess.run(cmd,stdout=output,stderr=subprocess.STDOUT,timeout=360)
            code=p.returncode
        except subprocess.TimeoutExpired:
            code=124
    results.append({'name':name,'command':cmd,'exit_code':code,'wall_seconds':time.perf_counter()-start})
    (ROOT/'evidence/02-components/suite-v2-execution.json').write_text(json.dumps(results,indent=2)+'\n')
    print('Finished',name,'exit',code,flush=True)
if any(r['exit_code']!=0 for r in results):raise SystemExit(1)
