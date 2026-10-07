"""Build the test-only native worker-release reproducer (not runtime code)."""
from pathlib import Path
import subprocess,sysconfig,hashlib,json
import torch
ROOT=Path(__file__).resolve().parents[1];source=ROOT/'coreml_port/native/cleanup_reproducer.mm';target=ROOT/'artifacts/04b-reliability/cleanup_reproducer.dylib'
cmd=['xcrun','clang++','-shared','-fPIC','-std=c++17','-fobjc-arc','-O2','-I'+sysconfig.get_paths()['include'],'-I'+str(Path(torch.__file__).parent/'include'),str(source),'-framework','Foundation','-undefined','dynamic_lookup','-o',str(target)];subprocess.run(cmd,check=True)
r={'command':cmd,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'binary_sha256':hashlib.sha256(target.read_bytes()).hexdigest()};(ROOT/'evidence/04b-reliability/reproducer-build.json').write_text(json.dumps(r,indent=2)+'\n')
