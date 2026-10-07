"""Sequential isolated conversions; stop at first failure and preserve logs."""
from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
for component in ['embedding',*map(str,range(1,12)),'head']:
    log=ROOT/'evidence/04-coreml'/f'convert-{component}.log'
    with log.open('w') as f:subprocess.run([sys.executable,str(ROOT/'scripts/convert_coreml.py'),component],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    print(component,'converted',flush=True)
