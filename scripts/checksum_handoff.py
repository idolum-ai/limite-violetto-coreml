"""Create and verify a whole-project handoff list, excluding volatile caches."""
from pathlib import Path
import os,hashlib,json,argparse
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--evidence-dir',default='evidence/04-coreml');a=p.parse_args();OUT=ROOT/a.evidence_dir;OUT.mkdir(exist_ok=True,parents=True)
skip={'.venv','.cache','.git','__pycache__'}
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
files=[]
for current,dirs,names in os.walk(ROOT):
    dirs[:]=[d for d in dirs if d not in skip]
    for name in names:
        if 'SHA256SUMS' in name or name in {'.DS_Store','checksum-verification.json','checksum-handoff.log'} or name.endswith('.pyc'):continue
        files.append(Path(current)/name)
entries=[(digest(p),str(p.relative_to(ROOT))) for p in sorted(files)]
(OUT/'SHA256SUMS').write_text(''.join(f'{h}  {p}\n' for h,p in entries))
failed=[p for h,p in entries if digest(ROOT/p)!=h]
result={'files_checked':len(entries),'failed':failed,'passed':not failed,'excluded':'virtualenv, caches, pyc, DS_Store, checksum lists, and this verification output'}
(OUT/'checksum-verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));raise SystemExit(bool(failed))
